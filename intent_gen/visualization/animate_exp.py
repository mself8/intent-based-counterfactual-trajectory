"""2차 실험 장면 애니메이션 — 실험 하나당 폴더 하나.

판 6개(1차 anim6와 같은 배치, IAD 자리에 이 실험):
  위: 실제 · PAUSA 등속(패스 X) · 1차 의도 적분(패스 X)
  아래: 이 실험 b=패스 O · 이 실험 b=패스 X · 1차 의도 적분(패스 O)
이 실험 판: 굵은 점과 꼬리 = 적분 궤적(결정론, PAUSA에 넣을 궤적), 가는 선 = 잔차 디퓨전 표본 20개.

    nice -n 19 /opt/conda/bin/python intent_gen/visualization/animate_exp.py --run exp1_bb_s0 --label '실험1' \
        --desc '의도 적분 궤적 + 잔차 디퓨전' --out_dir ./data/dfl/intent_gen/results/02_exp1_backbone_residual

장면 = 1차 8개(등속이 틀리는 장면) + 시험 경기 무작위 4개(passer 8 m 안에 수비수가 있는 장면, 씨앗 0).
모델 추론은 CPU(장면 하나라 빠름) — GPU 학습과 동시에 돌릴 수 있다.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation, font_manager  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # intent_gen/
from models.iad import DT, F, H, IADSoccer, prepare  # noqa: E402
from models.ours import Backbone, ResDiff, split_batch  # noqa: E402
from training.train_iad import KEYS  # noqa: E402
from visualization.animate_scene import BLUE, HOLD_END, HOLD_T, INK, PAST_SHOW, RED, TRAIL, draw_pitch  # noqa: E402

FIXED = [('pass_O', 'DFL-MAT-J03WMX', 43680, '쾰른–바이에른 217번 패스(학습 경기, 덱 44번 장면)', 'Away_40'),
         ('pass_O', 'DFL-MAT-J03WN1', 57969, '전반, Home_22 패스, 비르츠가 3.8 m에서 압박 중', 'Away_27'),
         ('pass_O', 'DFL-MAT-J03WN1', 73226, '전반, Home_20 패스, Away_25가 4.2 m에서 달려듦', 'Away_25'),
         ('pass_O', 'DFL-MAT-J03WN1', 50116, '전반, Away_9의 15 m 긴 패스, Home_20이 3.9 m', 'Home_20'),
         ('pass_X', 'DFL-MAT-J03WN1', 110762, '후반, 비르츠 공 유지, Home_10이 재껴진 뒤 추격', 'Home_10'),
         ('pass_X', 'DFL-MAT-J03WN1', 26218, '전반, 중원에서 비르츠 공 유지, Home_30이 4 m', 'Home_30'),
         ('pass_X', 'DFL-MAT-J03WN1', 146754, '후반, Home_7 공 유지, 비르츠가 3.3 m에서 수비', 'Away_27'),
         ('pass_X', 'DFL-MAT-J03WN1', 121313, '후반, 비르츠 공 유지, Home_31이 달아나다 돌아섬', 'Home_31')]
TEST_GAME = 'DFL-MAT-J03WN1'


def tensors(scene, b, dev):
    bt = {k: torch.as_tensor(scene[k][None].astype(np.int64) if scene[k].dtype == np.int8 else scene[k][None],
                             device=dev) for k in KEYS}
    bt['b'] = torch.full((1,), b, device=dev, dtype=torch.long)
    return bt


def random_scenes(sample_dir, n_each=2, seed=0):
    """시험 경기에서 passer 8 m 안에 수비수가 있는 장면을 b별로 무작위 n개."""
    d = np.load(os.path.join(sample_dir, f'{TEST_GAME}.npz'))
    p_t = d['pos'][:, :, H]
    dist = np.linalg.norm(p_t - p_t[:, :1], axis=-1)
    near = (d['mask'] & (d['team'] == 0) & (dist < 8)).any(1)
    fixed = {f for _, g, f, _, _ in FIXED if g == TEST_GAME}
    rng = np.random.default_rng(seed)
    out = []
    for kind, b in (('pass_O', 1), ('pass_X', 0)):
        idx = np.where(near & (d['b'] == b) & ~np.isin(d['frame'], list(fixed)))[0]
        for i in rng.choice(idx, n_each, replace=False):
            out.append((kind, TEST_GAME, int(d['frame'][i]), '무작위', None))
    return out


def load_scene(sample_dir, game, frame, kind):
    d = np.load(os.path.join(sample_dir, f'{game}.npz'))
    rows = np.where((d['frame'] == frame) & (d['b'] == (1 if kind == 'pass_O' else 0)))[0]
    assert len(rows), f'frame {frame} ({kind}) not in samples of {game}'
    return {k: d[k][rows[0]] for k in d.files}


@torch.no_grad()
def ref_rollout(scene, run_dir, dev):
    """1차 의도 적분(bdist_s0의 의도 예측기 출력만 적분)."""
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location=dev)
    m = IADSoccer(ck['cond']).to(dev)
    m.load_state_dict(ck['model'])
    m.eval()
    return {b: m.intention_rollout(prepare(tensors(scene, b, dev), ck['dv_std']))[0].cpu().numpy() for b in (1, 0)}


@torch.no_grad()
def exp_outputs(scene, run_dir, S, dev):
    g = torch.load(os.path.join(run_dir, 'g.pt'), map_location=dev)
    G = Backbone(g['feat'], g['ar']).to(dev)
    G.load_state_dict(g['model'])
    G.eval()
    dk = torch.load(os.path.join(run_dir, 'd.pt'), map_location=dev)
    D = ResDiff(dk['feat']).to(dev)
    D.load_state_dict(dk['model'])
    D.eval()
    bb, smp = {}, {}
    for b in (1, 0):
        sb = split_batch(tensors(scene, b, dev))
        traj = G.rollout(sb['P0'], sb['BL0'], sb['meta'])[0]
        torch.manual_seed(0)
        smp[b] = D.sample(sb['P0'], sb['BL0'], sb['meta'], traj, S=S)[0].cpu().numpy()
        bb[b] = traj[0].cpu().numpy()
    return bb, smp


def render(sc, panels, tag, out_dir, label, desc, fps=10, half_width=22.0, legend=None, focus=None):
    """panels = (제목, 미래 [22,25,2], 표본 또는 None, 공 표시[, 그 세계의 공 [25,2]]). 판이 3개 이하면 한 줄.
    focus = 주황 원으로 표시할 선수 번호(슬롯)."""
    pos, ball, mask = sc['pos'], sc['ball'], sc['mask']
    cx, cy = pos[0, H]
    half_w, half_h = half_width, half_width * 0.78
    cx = min(max(cx, -55.5 + half_w), 55.5 - half_w)
    cy = min(max(cy, -37 + half_h), 37 - half_h)
    x0, x1, y0, y1 = cx - half_w, cx + half_w, cy - half_h, cy + half_h

    nr = 1 if len(panels) <= 3 else 2
    fig, axes = plt.subplots(nr, 3, figsize=(19, 10.6 if nr == 2 else 6.3))
    fig.patch.set_facecolor('#FFFFFF')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9 if nr == 2 else 0.84, bottom=0.07 if nr == 2 else 0.1,
                        wspace=0.03, hspace=0.12)
    art = []
    cols = np.where(sc['team'] == 1, RED, BLUE)
    for ax, p in zip(axes.ravel(), panels):
        title, fut, samples, ball_mode = p[:4]
        ball_fut = p[4] if len(p) > 4 else None
        draw_pitch(ax, x0, x1, y0, y1)
        ax.set_title(title, fontsize=15, color=INK, loc='left', pad=6)
        trails = [ax.plot([], [], color=cols[j], lw=2.0, alpha=0.45, solid_capstyle='round')[0] for j in range(22)]
        n_s = samples.shape[0] if samples is not None else 0
        fan = [ax.plot([], [], color=cols[j % 22], lw=0.7, alpha=0.22)[0] for j in range(22 * n_s)]
        ghost = ax.scatter(np.zeros(22), np.zeros(22), s=150, facecolors='none', edgecolors='#9CA3AF',
                           linewidths=1.3, linestyle='--', zorder=3)
        dots = ax.scatter(np.zeros(22), np.zeros(22), s=150, c=cols, edgecolors='white', linewidths=1.2, zorder=5)
        ring = ax.scatter([0], [0], s=330, facecolors='none', edgecolors=INK, linewidths=1.8, zorder=6)
        bsolid = ax.scatter([0], [0], s=55, c=INK, zorder=7)
        bhollow = ax.scatter([0], [0], s=70, facecolors='white', edgecolors=INK, linewidths=1.6, zorder=7)
        fring = None if focus is None else ax.scatter([0], [0], s=430, facecolors='none', edgecolors='#F59E0B',
                                                      linewidths=2.6, zorder=6)
        art.append(dict(trails=trails, fan=fan, ghost=ghost, dots=dots, ring=ring, bsolid=bsolid, bhollow=bhollow,
                        fut=fut, samples=samples, ball_mode=ball_mode, ball_fut=ball_fut, fring=fring))
    fig.text(0.01, 0.965, f'{label} · {desc}', ha='left', va='center', fontsize=16, color=INK)
    stamp = fig.text(0.62 if nr == 2 else 0.85, 0.965, '', ha='center', va='center', fontsize=18, color=INK)
    fig.text(0.5, 0.025, legend or ('● 공   ○ 공은 생성하지 않음(패스 X 판은 passer 위치, 패스 O 판은 실제 공 위치)   ◎ passer   '
             '점선 원 = 같은 시각의 실제 위치   이 실험 판: 굵은 점 = 적분 궤적(PAUSA에 넣을 궤적), 가는 선 = 표본 20개'),
             ha='center', fontsize=12, color='#6B7280')
    idx_seq = list(range(H - PAST_SHOW, H)) + [H] * (HOLD_T + 1) + list(range(H + 1, H + F + 1)) + [H + F] * HOLD_END

    def frame_pos(a, k):
        return pos[:, k] if k <= H else a['fut'][:, k - H - 1]

    def update(n):
        k = idx_seq[n]
        stamp.set_text(f't = {(k - H) * DT:+.2f} s   ' + ('관측' if k <= H else '생성 구간'))
        for a in art:
            cur = frame_pos(a, k)
            a['dots'].set_offsets(np.where(mask[:, None], cur, np.nan))
            a['ring'].set_offsets(cur[:1])
            if a['fring'] is not None:
                a['fring'].set_offsets(cur[focus:focus + 1])
            ks = range(max(0, k - TRAIL), k + 1)
            for j, ln in enumerate(a['trails']):
                if mask[j]:
                    ln.set_data([frame_pos(a, kk)[j, 0] for kk in ks], [frame_pos(a, kk)[j, 1] for kk in ks])
            if a['ball_mode'] != 'real' and k > H:
                a['ghost'].set_offsets(np.where(mask[:, None], pos[:, k], np.nan))
            else:
                a['ghost'].set_offsets(np.full((22, 2), np.nan))
            if a['samples'] is not None:
                for s_ in range(a['samples'].shape[0]):
                    for j in range(22):
                        ln = a['fan'][s_ * 22 + j]
                        if mask[j] and k > H:
                            path = np.concatenate([pos[j, H:H + 1], a['samples'][s_, j, :k - H]])
                            ln.set_data(path[:, 0], path[:, 1])
                        else:
                            ln.set_data([], [])
            if k <= H or a['ball_mode'] == 'real':
                a['bsolid'].set_offsets(ball[k:k + 1])
                a['bhollow'].set_offsets(np.full((1, 2), np.nan))
            else:
                a['bsolid'].set_offsets(np.full((1, 2), np.nan))
                if a['ball_fut'] is not None:
                    bpos = a['ball_fut'][k - H - 1]
                else:
                    bpos = ball[k] if a['ball_mode'] == 'real_hollow' else cur[0]
                a['bhollow'].set_offsets(bpos[None])
        return []

    for k_snap, name in ((H, 't0'), (H + 12, 't+0.5'), (H + F, 't+1.0')):
        update(idx_seq.index(k_snap) if k_snap != H else PAST_SHOW)
        fig.savefig(os.path.join(out_dir, f'{tag}_{name}.png'), dpi=110)
    ani = animation.FuncAnimation(fig, update, frames=len(idx_seq), blit=False)
    gif = os.path.join(out_dir, f'{tag}.gif')
    ani.save(gif, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    ffs = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 '.venv/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*'))
    if ffs:
        subprocess.run([ffs[0], '-loglevel', 'error', '-y', '-i', gif, '-movflags', 'faststart', '-pix_fmt', 'yuv420p',
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-crf', '20',
                        os.path.join(out_dir, f'{tag}.mp4')], check=False)


def scene_stats(sc, outs, focus=None):
    """볼 수비수(1차 표와 같은 선수, 무작위 장면은 passer에게 가장 가까운 수비수)의 1초 접근(m, 음수 = 다가감)과 장면 전체 ADE."""
    pos, mask = sc['pos'], sc['mask']
    p_t = pos[:, H]
    dist = np.linalg.norm(p_t - p_t[:1], axis=-1)
    cand = np.where(mask & (sc['team'] == 0), dist, np.inf)
    j = int(np.argmin(cand)) if focus is None else int(np.where(sc['codes'] == focus)[0][0])
    res = {'defender': str(sc['codes'][j]), 'dist_t': round(float(dist[j]), 1)}
    for name, fut in outs.items():
        end = fut[:, -1]
        res[name] = round(float(np.linalg.norm(end[j] - end[0]) - dist[j]), 2)
        res[name + '_ADE'] = round(float(np.linalg.norm(fut - pos[:, H + 1:], axis=-1).mean(-1)[mask].mean()), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--label', required=True, help='짧은 이름, 판 제목에 씀 (예: 실험1)')
    ap.add_argument('--desc', required=True, help='그림 맨 위 설명')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ref_run', default='bdist_s0')
    ap.add_argument('--S', type=int, default=20)
    ap.add_argument('--only', type=int, default=-1, help='장면 번호 하나만(시험용)')
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--runs_dir', default='./data/dfl/intent_gen/runs')
    args = ap.parse_args()
    torch.set_num_threads(2)
    dev = 'cpu'
    for fp in ('/root/.fonts/NotoSansCJKkr-Regular.otf',):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = 'Noto Sans CJK KR'

    scenes = FIXED + random_scenes(args.sample_dir)
    if args.only >= 0:
        scenes = [scenes[args.only]]
    stats = []
    for kind, game, frame, note, focus in scenes:
        sc = load_scene(args.sample_dir, game, frame, kind)
        ref = ref_rollout(sc, os.path.join(args.runs_dir, args.ref_run), dev)
        bb, smp = exp_outputs(sc, os.path.join(args.runs_dir, args.run), args.S, dev)
        cv = sc['pos'][:, H:H + 1] + sc['vel_pausa'][:, None] * (np.arange(1, F + 1) * DT)[None, :, None]
        L = args.label
        panels = [('실제', sc['pos'][:, H + 1:], None, 'real'),
                  ('PAUSA 등속 · 패스 X', cv, None, 'carrier'),
                  ('1차 의도 적분 · 패스 X', ref[0], None, 'carrier'),
                  (f'{L} · b = 패스 O', bb[1], smp[1], 'real_hollow'),
                  (f'{L} · b = 패스 X', bb[0], smp[0], 'carrier'),
                  ('1차 의도 적분 · 패스 O', ref[1], None, 'real_hollow')]
        sub = os.path.join(args.out_dir, kind)
        os.makedirs(sub, exist_ok=True)
        tag = f'{game}_{frame}' + ('_rand' if note == '무작위' else '')
        render(sc, panels, tag, sub, L, args.desc)
        st = scene_stats(sc, {'real': sc['pos'][:, H + 1:], 'cv': cv, 'ref_X': ref[0], 'ref_O': ref[1],
                              'exp_X': bb[0], 'exp_O': bb[1], 'exp_X_mean': smp[0].mean(0),
                              'exp_O_mean': smp[1].mean(0)}, focus)
        stats.append({'kind': kind, 'file': tag, 'note': note, **st})
        print(json.dumps(stats[-1], ensure_ascii=False), flush=True)
    if args.only < 0:
        with open(os.path.join(args.out_dir, 'scene_stats.json'), 'w') as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
