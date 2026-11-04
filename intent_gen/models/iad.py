"""IAD 축구판 — 의도 조건 1초 장면 생성기.

IAD(Liu et al., AAAI 2026) 구조를 따른다: Motion Encoder -> F_obs,
단기 의도(가속도 방향 theta·크기 r, residual polar) + 1초 뒤 도착점 후보 L개와 확률,
soft mask로 섞은 조건, Noise Estimator + RefineNet, DDIM 5번 건너뛰기.

축구용으로 바꾼 점:
  - 22명을 한 장면으로 함께 생성한다(선수끼리 어텐션, 공은 문맥 토큰).
  - 가속도 크기 변화를 log r 잔차로 예측한다(IAD의 softplus는 늘기만 한다).
  - 추론 때 도착점은 확률이 가장 높은 후보를 쓴다(논문 설명대로, 공식 코드는 정답 최근접).
  - 생성 대상은 마지막 관측 속도 대비 25Hz 속도 변화이고, 적분해서 위치를 만든다.
  - cond='b_dist': 공 상태 b(1 = 패스 O, 0 = 패스 X)와 소유자와의 거리를 F_obs에 붙인다.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as Fn

H, F, DT = 50, int(os.environ.get('FUT_FRAMES', 25)), 0.04
HIST_IDX = list(range(0, H + 1, 5))   # 11 points at 5 Hz, t-2.0 s .. t
INT_STEPS = 5                          # intention every 0.2 s over the next 1 s
V_SCALE, A_SCALE, R_EPS = 5.0, 5.0, 0.1
R_MAX, R_GT_CLIP = 50.0, 20.0          # m/s^2; tracking noise can still give spikes after the 12 m/s filter
N_P = 22


def _polar(a):
    return torch.atan2(a[..., 1], a[..., 0]), a.norm(dim=-1)


def prepare(batch, dv_std, flip_y=False):
    """npz 배치(텐서) -> 모델 입력·정답. 모든 값은 과거(t까지)만으로 입력을 만든다."""
    # padded slots are NaN; masked keys still leak NaN through attention (0 * NaN), so zero-fill
    pos = torch.nan_to_num(batch['pos'].clone())
    ball = torch.nan_to_num(batch['ball'].clone())
    if flip_y:
        pos[..., 1] *= -1
        ball[..., 1] *= -1
    B = pos.shape[0]
    hist = pos[:, :, HIST_IDX]                                   # [B,22,11,2]
    v_h = torch.diff(hist, dim=2) / 0.2
    v_h = torch.cat([v_h[:, :, :1], v_h], 2)
    bh = ball[:, HIST_IDX]
    bv = torch.diff(bh, dim=1) / 0.2
    bv = torch.cat([bv[:, :1], bv], 1)
    scale = pos.new_tensor([52.5, 34.0])
    p_t = pos[:, :, H]
    v_t = (p_t - pos[:, :, H - 3]) / (3 * DT)                     # past-only velocity at t
    a0 = ((pos[:, :, H] - pos[:, :, H - 5]) - (pos[:, :, H - 5] - pos[:, :, H - 10])) / 0.2 ** 2
    th0, r0 = _polar(a0)
    team = batch['team'].long()
    is_car = torch.zeros_like(batch['mask'])
    is_car[:, 0] = True
    static = torch.cat([team[..., None].float(), is_car[..., None].float(), batch['gk'][..., None].float(),
                        v_t / V_SCALE, torch.cos(th0)[..., None], torch.sin(th0)[..., None],
                        (r0 / A_SCALE)[..., None]], -1)
    dist = (p_t - p_t[:, :1]).norm(dim=-1)                       # to carrier at t
    b = batch['b'].float()[:, None].expand(B, N_P)
    ld = torch.log1p(dist) / 3
    bd = torch.stack([b, dist / 20, b * dist / 20, ld, b * ld], -1)

    f = {'hist': torch.cat([hist / scale, v_h / V_SCALE], -1),
         'ball_hist': torch.cat([bh / scale, bv / V_SCALE], -1),
         'static': static, 'team': team, 'mask': batch['mask'], 'b': batch['b'].long(), 'bd': bd,
         'p_t': p_t, 'v_t': v_t, 'th0': th0, 'r0': r0, 'cv_end': v_t * F * DT,
         'v5_0': (pos[:, :, H] - pos[:, :, H - 5]) / 0.2}      # mean velocity over the last 0.2 s
    if pos.shape[2] > H + 1:   # targets
        fut = pos[:, :, H:]                                          # t .. t+25
        v_f = torch.diff(fut, dim=2) / DT                            # [B,22,25,2]
        f['x0'] = (v_f - v_t[:, :, None]) / dv_std
        p5 = pos[:, :, list(range(H - 5, H + F + 1, 5))]            # t-0.2 .. t+1.0
        v5 = torch.diff(p5, dim=2) / 0.2                             # [B,22,6,2]
        a5 = torch.diff(v5, dim=2) / 0.2                             # [B,22,5,2]
        f['th_gt'], f['r_gt'] = _polar(a5)
        f['e_gt'] = pos[:, :, H + F] - p_t
    return f


def _layer(d, heads):
    return nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)


def _mlp(i, d, o):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))


class StepEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.net = _mlp(d, d, d)

    def forward(self, k):
        half = self.d // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=k.device) / half)
        a = k.float()[:, None] * freq[None]
        return self.net(torch.cat([a.sin(), a.cos()], -1))


class IADSoccer(nn.Module):
    def __init__(self, cond='past', d=128, heads=4, enc_layers=2, den_layers=4, L=5, K=100):
        super().__init__()
        assert cond in ('past', 'b_dist')
        self.cond, self.L, self.K = cond, L, K
        # Motion Encoder
        self.step_in = nn.Linear(4, d)
        self.time_pos = nn.Parameter(torch.zeros(len(HIST_IDX), d))
        self.temporal = nn.TransformerEncoder(_layer(d, heads), 1)
        self.static_in = nn.Linear(8, d)
        self.type_emb = nn.Embedding(3, d)        # 0 defender, 1 attacker, 2 ball
        if cond == 'b_dist':
            self.bd_in = _mlp(5, d, d)
            self.b_ball = nn.Embedding(2, d)
        self.social = nn.TransformerEncoder(_layer(d, heads), enc_layers)
        # Intention Estimator (residual polar)
        self.intent = _mlp(d, d, INT_STEPS * 3)
        # Endpoint Estimator
        self.queries = nn.Parameter(torch.randn(L, d) * 0.02)
        self.ep_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ep_out = _mlp(d, d, 3)
        # guidance + soft mask fusion
        self.g_s = _mlp(INT_STEPS * 3, d, d)
        self.g_l = _mlp(2, d, d)
        self.mask_lin = nn.ModuleList([nn.Linear(d, 1) for _ in range(3)])
        self.psi = nn.Parameter(torch.zeros(3, d))
        self.fuse = nn.Linear(3 * d, d)
        # Noise Estimator + RefineNet
        self.k_emb = StepEmb(d)
        self.x_in = nn.Linear(F * 2, d)
        self.denoiser = nn.TransformerEncoder(_layer(d, heads), den_layers)
        self.eps_out = nn.Linear(d, F * 2)
        self.refine = nn.Sequential(nn.Linear(F * 4 + 2 * d, 2 * d), nn.GELU(),
                                    nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, F * 2))
        betas = torch.linspace(1e-4, 0.05, K)
        self.register_buffer('ab', torch.cumprod(1 - betas, 0))

    # ---- conditions -------------------------------------------------------------------------
    def encode(self, f):
        B = f['hist'].shape[0]
        tok = self.step_in(f['hist']) + self.time_pos
        tok = self.temporal(tok.flatten(0, 1)).view(B, N_P, len(HIST_IDX), -1)
        summ = tok[:, :, -1] + self.static_in(f['static']) + self.type_emb(f['team'])
        bt = self.temporal(self.step_in(f['ball_hist']) + self.time_pos)[:, -1] + self.type_emb.weight[2]
        if self.cond == 'b_dist':   # F_obs = [MotionEncoder(X), b, distance]
            summ = summ + self.bd_in(f['bd'])
            bt = bt + self.b_ball(f['b'])
        x = torch.cat([summ, bt[:, None]], 1)
        pad = torch.cat([~f['mask'], f['mask'].new_zeros(B, 1)], 1)
        x = self.social(x, src_key_padding_mask=pad)
        return x[:, :N_P], x[:, N_P:], tok

    def intention(self, fo, f):
        o = self.intent(fo).view(*fo.shape[:2], INT_STEPS, 3)
        dth = torch.atan2(o[..., 1], 1 + o[..., 0])
        th = f['th0'][..., None] + torch.cumsum(dth, -1)
        lr = torch.log(f['r0'] + R_EPS)[..., None] + torch.cumsum(o[..., 2], -1)
        return th, torch.exp(lr.clamp(max=math.log(R_MAX))) - R_EPS

    def endpoints(self, fo, tok, f):
        B = fo.shape[0]
        q = self.queries[None] + fo.reshape(B * N_P, 1, -1)
        kv = torch.cat([tok.flatten(0, 1), fo.reshape(B * N_P, 1, -1)], 1)
        h, _ = self.ep_attn(q, kv, kv)
        o = self.ep_out(h).view(B, N_P, self.L, 3)
        return f['cv_end'][:, :, None] + 2.0 * o[..., :2], o[..., 2]

    def guidance(self, fo, th, r, e_sel):
        gs = self.g_s(torch.cat([th.cos(), th.sin(), r / A_SCALE], -1))
        gl = self.g_l(e_sel / 5.0)
        out, masks = [], []
        for m, g in enumerate((fo, gs, gl)):
            M = torch.sigmoid(self.mask_lin[m](g))
            out.append((1 - M) * g + M * self.psi[m])
            masks.append(M)
        return self.fuse(torch.cat(out, -1)), torch.cat(masks, -1)

    def conditions(self, f, e_gt=None):
        fo, ball_ctx, tok = self.encode(f)
        th, r = self.intention(fo, f)
        ends, logits = self.endpoints(fo, tok, f)
        if e_gt is None:
            idx = logits.argmax(-1)
        else:
            idx = (ends - e_gt[:, :, None]).norm(dim=-1).argmin(-1)
        e_sel = ends.gather(2, idx[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
        c, masks = self.guidance(fo, th, r, e_sel)
        return dict(c=c, ball_ctx=ball_ctx, th=th, r=r, ends=ends, logits=logits, idx=idx, masks=masks)

    def intention_rollout(self, f, cd=None):
        """예측한 단기 의도(0.2초마다 가속도)만 적분한 결정론 궤적 -> [B,22,F,2] (m)."""
        if cd is None:
            cd = self.conditions(f)
        a = torch.stack([cd['r'] * torch.cos(cd['th']), cd['r'] * torch.sin(cd['th'])], -1)
        v, p, out = f['v5_0'], f['p_t'], []
        for s in range(INT_STEPS):
            v = v + a[:, :, s] * (F * DT / INT_STEPS)
            for _ in range(F // INT_STEPS):
                p = p + v * DT
                out.append(p)
        return torch.stack(out, 2)

    # ---- noise ----------------------------------------------------------------------------
    def eps(self, xk, k, c, ball_ctx, mask):
        ke = self.k_emb(k)[:, None]
        h = self.x_in(xk.flatten(-2)) + c + ke
        h = torch.cat([h, ball_ctx], 1)
        pad = torch.cat([~mask, mask.new_zeros(mask.shape[0], 1)], 1)
        h = self.denoiser(h, src_key_padding_mask=pad)[:, :N_P]
        e = self.eps_out(h)
        de = self.refine(torch.cat([xk.flatten(-2), e, c, ke.expand_as(c)], -1))
        return (e + de).view_as(xk)

    def loss(self, f):
        m = f['mask']
        cd = self.conditions(f, e_gt=f['e_gt'])
        w = (f['r_gt'] > 0.5).float() * m[..., None]
        l_th = ((1 - torch.cos(cd['th'] - f['th_gt'])) * w).sum() / w.sum().clamp(min=1)
        l_r = (((cd['r'] - f['r_gt'].clamp(max=R_GT_CLIP)) / A_SCALE) ** 2)[m].mean()
        dist = (cd['ends'] - f['e_gt'][:, :, None]).norm(dim=-1)
        l_e = dist.min(-1).values[m].mean()
        l_p = Fn.cross_entropy(cd['logits'][m], cd['idx'][m])
        B = m.shape[0]
        k = torch.randint(0, self.K, (B,), device=m.device)
        noise = torch.randn_like(f['x0'])
        ab = self.ab[k][:, None, None, None]
        xk = ab.sqrt() * f['x0'] + (1 - ab).sqrt() * noise
        e_hat = self.eps(xk, k, cd['c'], cd['ball_ctx'], m)
        l_dif = ((e_hat - noise) ** 2).mean(dim=(-1, -2))[m].mean()
        parts = {'theta': l_th, 'r': l_r, 'end': l_e, 'prob': l_p, 'diff': l_dif}
        return sum(parts.values()), parts

    @torch.no_grad()
    def sample(self, f, dv_std, S=20, steps=5):
        """-> 미래 위치 [B,S,22,25,2] (m)."""
        cd = self.conditions(f)
        B = f['mask'].shape[0]
        rep = lambda x: x.repeat_interleave(S, 0)  # noqa: E731
        c, bc, m = rep(cd['c']), rep(cd['ball_ctx']), rep(f['mask'])
        x = torch.randn(B * S, N_P, F, 2, device=c.device)
        ks = list(range(self.K - 1, -1, -(self.K // steps)))
        for i, k in enumerate(ks):
            kk = torch.full((B * S,), k, device=c.device, dtype=torch.long)
            e = self.eps(x, kk, c, bc, m)
            ab = self.ab[k]
            x0 = (x - (1 - ab).sqrt() * e) / ab.sqrt()
            if i == len(ks) - 1:
                x = x0
            else:
                ab_n = self.ab[ks[i + 1]]
                x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * e
        v = rep(f['v_t'])[:, :, None] + x * dv_std
        pos = rep(f['p_t'])[:, :, None] + torch.cumsum(v * DT, dim=2)
        return pos.view(B, S, N_P, F, 2), cd
