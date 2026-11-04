"""2차 라운드 — 의도 적분 궤적(결정론) + 잔차 디퓨전.

1차 진단(10-31~11-01): 의도 예측기 출력만 적분한 궤적이 등속·IAD를 모두 이겼고, IAD 디퓨전은 의도를 버렸다.
그래서 의도 적분 궤적을 기본으로 두고, 디퓨전은 '실제 - 적분 궤적' 잔차만 만든다(①).

적분 궤적 Backbone: 0.2초마다의 가속도(방향 theta, 크기 r)를 예측해 적분한 1초 궤적.
  ar=False  t에서 5단계 의도를 한 번에 예측(1차 의도 적분과 같은 방식).
  ar=True   0.2초 의도 하나 예측 -> 적분 -> 바뀐 장면을 다시 인코딩, 5번 반복(③).
  feat='bdist'  1차와 같은 입력(과거 2초, b, 소유자 거리).
  feat='rel'    + passer 기준 상대 좌표·속도, 과거 가속도의 passer 방향 성분(phi), 재껴짐·접근 속도(②).
  feat='rel_terrain'  rel + 골대·측면선까지 거리. feat='rel_bd3'  rel에서 b·거리 입력을 b, 거리, 단계 3개로 축소.
  feat='rel_lean'  rel_bd3에서 재껴짐까지 뺀 입력.
잔차 ResDiff: 0.2초 마디 5개에서 '실제 - 적분 궤적' 위치 잔차를 디퓨전으로 생성(적분 궤적은 고정). 마디 사이는 직선.

공 조건: t까지는 실제 공. 자기회귀에서 t 뒤의 공은 b=0(패스 X)이면 passer 위치, b=1(패스 O)이면 t 위치에 고정.
"""
import math
import os

import torch
import torch.nn as nn

H, F, DT = 50, int(os.environ.get('FUT_FRAMES', 25)), 0.04
HIST_IDX = list(range(0, H + 1, 5))    # 11 knots at 5 Hz, t-2.0 s .. t
STEP = 5
INT_STEPS = F // STEP                   # intentions of 0.2 s (= 5 frames) over the future window
V_SCALE, A_SCALE, R_EPS = 5.0, 5.0, 0.1
R_MAX, R_GT_CLIP = 50.0, 20.0
N_P = 22
SCALE = (52.5, 34.0)
HIST_DIM = {'bdist': 4, 'rel': 11, 'rel_terrain': 11, 'rel_bd3': 11, 'rel_lean': 11}
STATIC_DIM = {'bdist': 8, 'rel': 13, 'rel_terrain': 15, 'rel_bd3': 13, 'rel_lean': 12}
BD_DIM = {'bdist': 6, 'rel': 6, 'rel_terrain': 6, 'rel_bd3': 3, 'rel_lean': 3}


def _polar(a):
    return torch.atan2(a[..., 1], a[..., 0]), a.norm(dim=-1)


def window_feats(P, BL, meta, feat, s=0):
    """P [B,22,H+1,2] = 현재 프레임 c에서 끝나는 2초 창의 위치, BL [B,H+1,2] = 같은 창의 공 조건.
    meta = mask, team, gk, b. s = 자기회귀 단계(t 뒤 0.2*s 초)."""
    B = P.shape[0]
    hist = P[:, :, HIST_IDX]
    v_h = torch.diff(hist, dim=2) / 0.2
    v_h = torch.cat([v_h[:, :, :1], v_h], 2)
    bh = BL[:, HIST_IDX]
    bv = torch.diff(bh, dim=1) / 0.2
    bv = torch.cat([bv[:, :1], bv], 1)
    scale = P.new_tensor(SCALE)
    p_t = P[:, :, H]
    v_t = (p_t - P[:, :, H - 3]) / (3 * DT)
    a0 = ((P[:, :, H] - P[:, :, H - 5]) - (P[:, :, H - 5] - P[:, :, H - 10])) / 0.2 ** 2
    th0, r0 = _polar(a0)
    team = meta['team'].long()
    is_car = torch.zeros_like(meta['mask'])
    is_car[:, 0] = True
    static = [team[..., None].float(), is_car[..., None].float(), meta['gk'][..., None].float(),
              v_t / V_SCALE, torch.cos(th0)[..., None], torch.sin(th0)[..., None], (r0 / A_SCALE)[..., None]]
    dist = (p_t - p_t[:, :1]).norm(dim=-1)
    b = meta['b'].float()[:, None].expand(B, N_P)
    ld = torch.log1p(dist) / 3
    bd = torch.stack([b, dist / 20, b * dist / 20, ld, b * ld, torch.full_like(b, s / INT_STEPS)], -1)
    if feat in ('rel_bd3', 'rel_lean'):
        bd = bd[..., [0, 1, 5]]
    hist_f = [hist / scale, v_h / V_SCALE]
    if feat != 'bdist':
        rel = hist - hist[:, :1]                                   # to the passer, at every knot
        relv = torch.diff(rel, dim=2) / 0.2
        relv = torch.cat([relv[:, :, :1], relv], 2)
        a_h = torch.diff(v_h, dim=2) / 0.2
        a_h = torch.cat([a_h[:, :, :1], a_h], 2)
        u = -rel / rel.norm(dim=-1, keepdim=True).clamp(min=0.5)   # unit vector to the passer (0 for the passer)
        a_par = (a_h * u).sum(-1, keepdim=True)
        a_perp = (u[..., :1] * a_h[..., 1:] - u[..., 1:] * a_h[..., :1])
        ra = a_h.norm(dim=-1, keepdim=True)
        hist_f += [rel / 20.0, relv / V_SCALE, a_par / (ra + 1.0), a_perp / (ra + 1.0), ra / A_SCALE]
        x_rel = p_t[..., 0] - p_t[:, :1, 0]                         # + = nearer the goal being attacked (+x)
        beaten = ((team == 0) & (x_rel < -1.0)).float()
        d_prev = (P[:, :, H - 5] - P[:, :1, H - 5]).norm(dim=-1)
        closing = (d_prev - dist) / 0.2 / V_SCALE
        u_t = -(p_t - p_t[:, :1]) / dist[..., None].clamp(min=0.5)
        static += [(x_rel / 20)[..., None]] + ([] if feat == 'rel_lean' else [beaten[..., None]]) + [closing[..., None], u_t]
        if feat == 'rel_terrain':                                   # 공격받는 골대(+x)까지 거리, 가까운 측면선까지 거리
            d_goal = (p_t - p_t.new_tensor((SCALE[0], 0.0))).norm(dim=-1)
            static += [(d_goal / 50)[..., None], ((SCALE[1] - p_t[..., 1].abs()) / SCALE[1])[..., None]]
    return {'hist': torch.cat(hist_f, -1), 'ball_hist': torch.cat([bh / scale, bv / V_SCALE], -1),
            'static': torch.cat(static, -1), 'team': team, 'mask': meta['mask'], 'b': meta['b'].long(), 'bd': bd,
            'p_t': p_t, 'th0': th0, 'r0': r0, 'v5_0': (P[:, :, H] - P[:, :, H - 5]) / 0.2, 'dist': dist}


def split_batch(bt, flip=False):
    """npz 배치(텐서) -> 과거 창·공·메타·정답. 패딩 슬롯 NaN은 0으로(어텐션에 0*NaN이 새는 것 방지)."""
    pos = torch.nan_to_num(bt['pos'].clone())
    ball = torch.nan_to_num(bt['ball'].clone())
    if flip:
        pos[..., 1] *= -1
        ball[..., 1] *= -1
    meta = {'mask': bt['mask'], 'team': bt['team'], 'gk': bt['gk'], 'b': bt['b'].long()}
    out = {'P0': pos[:, :, :H + 1], 'BL0': ball[:, :H + 1], 'meta': meta}
    if pos.shape[2] > H + 1:
        out['fut'] = pos[:, :, H + 1:]
        p5 = pos[:, :, list(range(H - 5, H + F + 1, 5))]
        a5 = torch.diff(torch.diff(p5, dim=2) / 0.2, dim=2) / 0.2   # [B,22,5,2]
        out['th_gt'], out['r_gt'] = _polar(a5)
    return out


def knots_to_frames(p_t, knots):
    """p_t [...,22,2], knots [...,22,5,2] (t+0.2 .. t+1.0) -> [...,22,25,2], 마디 사이는 직선."""
    pts = torch.cat([p_t[..., None, :], knots], -2)
    w = torch.arange(1, STEP + 1, device=p_t.device, dtype=p_t.dtype) / STEP
    seg = pts[..., :-1, None, :] + (pts[..., 1:, None, :] - pts[..., :-1, None, :]) * w[:, None]
    return seg.flatten(-3, -2)


def _layer(d, heads):
    return nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)


def _mlp(i, d, o):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))


class Encoder(nn.Module):
    """선수별 과거 창 -> 시간 트랜스포머 요약 + 정적·b·거리 -> 22명+공 사회 트랜스포머."""

    def __init__(self, feat, d=128, heads=4, layers=2):
        super().__init__()
        self.step_in = nn.Linear(HIST_DIM[feat], d)
        self.ball_in = nn.Linear(4, d)
        self.time_pos = nn.Parameter(torch.zeros(len(HIST_IDX), d))
        self.temporal = nn.TransformerEncoder(_layer(d, heads), 1)
        self.static_in = nn.Linear(STATIC_DIM[feat], d)
        self.type_emb = nn.Embedding(3, d)        # 0 defender, 1 attacker, 2 ball
        self.bd_in = _mlp(BD_DIM[feat], d, d)
        self.b_ball = nn.Embedding(2, d)
        self.step_emb = nn.Embedding(INT_STEPS, d)
        self.social = nn.TransformerEncoder(_layer(d, heads), layers)

    def forward(self, f, s=0):
        B = f['hist'].shape[0]
        tok = self.step_in(f['hist']) + self.time_pos
        tok = self.temporal(tok.flatten(0, 1)).view(B, N_P, len(HIST_IDX), -1)
        se = self.step_emb.weight[s]
        summ = tok[:, :, -1] + self.static_in(f['static']) + self.type_emb(f['team']) + self.bd_in(f['bd']) + se
        bt = self.temporal(self.ball_in(f['ball_hist']) + self.time_pos)[:, -1] + self.type_emb.weight[2] + \
            self.b_ball(f['b']) + se
        x = torch.cat([summ, bt[:, None]], 1)
        pad = torch.cat([~f['mask'], f['mask'].new_zeros(B, 1)], 1)
        x = self.social(x, src_key_padding_mask=pad)
        return x[:, :N_P], x[:, N_P:]


class Backbone(nn.Module):
    """의도(0.2초마다 가속도)를 예측해 적분하는 결정론 1초 궤적."""

    def __init__(self, feat='bdist', ar=False, d=128, heads=4, layers=2):
        super().__init__()
        self.feat, self.ar = feat, ar
        self.enc = Encoder(feat, d, heads, layers)
        self.head = _mlp(d, d, (1 if ar else INT_STEPS) * 3)

    @staticmethod
    def _accel(o, th0, r0):
        """residual polar (IAD): 직전 가속도 방향·log 크기에서 누적 변화."""
        th = th0[..., None] + torch.cumsum(torch.atan2(o[..., 1], 1 + o[..., 0]), -1)
        lr = torch.log(r0.clamp(min=0) + R_EPS)[..., None] + torch.cumsum(o[..., 2], -1)
        return th, torch.exp(lr.clamp(max=math.log(R_MAX))) - R_EPS

    def rollout(self, P0, BL0, meta):
        """-> traj [B,22,25,2] (m), th·r [B,22,5]."""
        B = P0.shape[0]
        if not self.ar:
            f = window_feats(P0, BL0, meta, self.feat, 0)
            fo, _ = self.enc(f, 0)
            th, r = self._accel(self.head(fo).view(B, N_P, INT_STEPS, 3), f['th0'], f['r0'])
            a = torch.stack([r * torch.cos(th), r * torch.sin(th)], -1)
            v = f['v5_0'][:, :, None] + torch.cumsum(a, 2) * (STEP * DT)          # velocity in each 0.2 s
            knots = f['p_t'][:, :, None] + torch.cumsum(v * (STEP * DT), 2)
            return knots_to_frames(f['p_t'], knots), th, r
        P, BL = P0, BL0
        hold = (meta['b'] == 0)[:, None, None]
        frozen = BL0[:, -1:].expand(B, STEP, 2)
        segs, ths, rs = [], [], []
        v = p = None
        for s in range(INT_STEPS):
            f = window_feats(P.detach(), BL.detach(), meta, self.feat, s)
            if s == 0:
                v, p = f['v5_0'], f['p_t']
            fo, _ = self.enc(f, s)
            th, r = self._accel(self.head(fo).view(B, N_P, 1, 3), f['th0'].detach(), f['r0'].detach())
            a = torch.stack([r[..., 0] * torch.cos(th[..., 0]), r[..., 0] * torch.sin(th[..., 0])], -1)
            v = v + a * (STEP * DT)
            seg = p[:, :, None] + v[:, :, None] * (DT * torch.arange(1, STEP + 1, device=P0.device,
                                                                     dtype=P0.dtype))[:, None]
            p = seg[:, :, -1]
            segs.append(seg)
            ths.append(th[..., 0])
            rs.append(r[..., 0])
            P = torch.cat([P[:, :, STEP:], seg.detach()], 2)
            ball_seg = torch.where(hold, seg[:, 0].detach(), frozen)
            BL = torch.cat([BL[:, STEP:], ball_seg], 1)
        return torch.cat(segs, 2), torch.stack(ths, -1), torch.stack(rs, -1)

    def loss(self, bt, weighted=False, lam_traj=1.0):
        traj, th, r = self.rollout(bt['P0'], bt['BL0'], bt['meta'])
        m = bt['meta']['mask'].float()
        dist = (bt['P0'][:, :, H] - bt['P0'][:, :1, H]).norm(dim=-1)
        if weighted:   # ②: passer 10 m 안 수비수 x3, 큰 가속 최대 x3
            w_p = 1 + 2 * ((bt['meta']['team'] == 0) & (dist < 10)).float()
            w_a = (1 + bt['r_gt'].clamp(max=R_GT_CLIP) / 5).clamp(max=3)
        else:
            w_p, w_a = torch.ones_like(m), torch.ones_like(bt['r_gt'])
        w_th = (bt['r_gt'] > 0.5).float() * (m * w_p)[..., None] * w_a
        l_th = ((1 - torch.cos(th - bt['th_gt'])) * w_th).sum() / w_th.sum().clamp(min=1)
        w_r = (m * w_p)[..., None] * w_a
        l_r = ((((r - bt['r_gt'].clamp(max=R_GT_CLIP)) / A_SCALE) ** 2) * w_r).sum() / w_r.sum().clamp(min=1)
        err = (traj - bt['fut']).norm(dim=-1).mean(-1)
        l_traj = (err * m * w_p).sum() / (m * w_p).sum().clamp(min=1)
        parts = {'theta': l_th, 'r': l_r, 'traj': l_traj}
        return l_th + l_r + lam_traj * l_traj, parts


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


class ResDiff(nn.Module):
    """'실제 - 적분 궤적'를 0.2초 마디 5개(10차원)에서 생성하는 디퓨전. 조건 = 장면 인코딩 + 적분 궤적 마디."""

    def __init__(self, feat='bdist', d=128, heads=4, enc_layers=2, den_layers=3, K=100):
        super().__init__()
        self.feat, self.K = feat, K
        self.enc = Encoder(feat, d, heads, enc_layers)
        self.bb_in = _mlp(INT_STEPS * 2, d, d)
        self.x_in = nn.Linear(INT_STEPS * 2, d)
        self.k_emb = StepEmb(d)
        self.den = nn.TransformerEncoder(_layer(d, heads), den_layers)
        self.out = nn.Linear(d, INT_STEPS * 2)
        betas = torch.linspace(1e-4, 0.2, K)
        self.register_buffer('ab', torch.cumprod(1 - betas, 0))
        self.register_buffer('res_std', torch.ones(INT_STEPS))

    def _cond(self, P0, BL0, meta, bb_traj):
        f = window_feats(P0, BL0, meta, self.feat, 0)
        fo, bt = self.enc(f, 0)
        bbk = (bb_traj[:, :, STEP - 1::STEP] - f['p_t'][:, :, None]) / 5.0
        return fo + self.bb_in(bbk.flatten(-2)), bt, f

    def eps(self, xk, k, c, bt, mask):
        h = self.x_in(xk.flatten(-2)) + c + self.k_emb(k)[:, None]
        h = torch.cat([h, bt], 1)
        pad = torch.cat([~mask, mask.new_zeros(mask.shape[0], 1)], 1)
        return self.out(self.den(h, src_key_padding_mask=pad)[:, :N_P]).view_as(xk)

    def residual(self, bb_traj, fut):
        return (fut[:, :, STEP - 1::STEP] - bb_traj[:, :, STEP - 1::STEP]) / self.res_std[:, None]

    def loss(self, bt, bb_traj):
        c, ball_tok, _ = self._cond(bt['P0'], bt['BL0'], bt['meta'], bb_traj)
        x0 = self.residual(bb_traj, bt['fut'])
        m = bt['meta']['mask']
        k = torch.randint(0, self.K, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.ab[k][:, None, None, None]
        e_hat = self.eps(ab.sqrt() * x0 + (1 - ab).sqrt() * noise, k, c, ball_tok, m)
        return ((e_hat - noise) ** 2).mean(dim=(-1, -2))[m].mean()

    @torch.no_grad()
    def sample(self, P0, BL0, meta, bb_traj, S=20, steps=10):
        """-> [B,S,22,25,2] (m). 적분 궤적 마디 + 생성 잔차, 마디 사이 직선."""
        c, ball_tok, f = self._cond(P0, BL0, meta, bb_traj)
        B = c.shape[0]
        rep = lambda x: x.repeat_interleave(S, 0)  # noqa: E731
        c, ball_tok, m = rep(c), rep(ball_tok), rep(meta['mask'])
        x = torch.randn(B * S, N_P, INT_STEPS, 2, device=c.device)
        ks = list(range(self.K - 1, -1, -(self.K // steps)))
        for i, k in enumerate(ks):
            kk = torch.full((B * S,), k, device=c.device, dtype=torch.long)
            e = self.eps(x, kk, c, ball_tok, m)
            ab = self.ab[k]
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-5, 5)   # residual in std units
            if i == len(ks) - 1:
                x = x0
            else:
                ab_n = self.ab[ks[i + 1]]
                x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * e
        knots = rep(bb_traj[:, :, STEP - 1::STEP]) + x * self.res_std[:, None]
        return knots_to_frames(rep(f['p_t']), knots).view(B, S, N_P, F, 2)
