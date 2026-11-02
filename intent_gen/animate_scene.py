"""한 장면의 1초를 네 판으로 나란히 애니메이션: 실제 / PAUSA 등속(패스 X) / IAD b=패스 O / IAD b=패스 X.

    CUDA_VISIBLE_DEVICES=0 nice -n 19 /opt/conda/bin/python intent_gen/animate_scene.py \
        --game DFL-MAT-J03WMX --frame 43680 --run bdist_s0

frame = 기준 프레임 t(= PAUSA pass_frame - 1). 표본(build_dataset.py)에 있는 장면만 그린다.
"""
import argparse
import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation, font_manager  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DT, F, H, IADSoccer, prepare  # noqa: E402
from train import KEYS  # noqa: E402

RED, BLUE, INK, LINE = '#C8372D', '#1F5FBF', '#1A1A1A', '#B8BEC6'
PAST_SHOW, TRAIL = 37, 25          # 1.5 s of observed past, 1 s trail
HOLD_T, HOLD_END = 8, 15           # extra frames to pause at t and at the end


def load_scene(sample_dir, game, frame):
    d = np.load(os.path.join(sample_dir, f'{game}.npz'))
    rows = np.where((d['frame'] == frame) & (d['b'] == 1))[0]
    if len(rows) == 0:
        rows = np.where(d['frame'] == frame)[0]
    assert len(rows), f'frame {frame} not in samples of {game}'
    i = rows[0]
    return {k: d[k][i] for k in d.files}


@torch.no_grad()
def iad_samples(scene, run_dir, S, dev='cuda'):
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location=dev)
    m = IADSoccer(ck['cond']).to(dev)
    m.load_state_dict(ck['model'])
    m.eval()
    out, roll = {}, {}
    for b in (1, 0):
        bt = {k: torch.as_tensor(scene[k][None].astype(np.int64) if scene[k].dtype == np.int8 else scene[k][None],
                                 device=dev) for k in KEYS}
        bt['b'] = torch.full((1,), b, device=dev, dtype=torch.long)
        torch.manual_seed(0)
        f = prepare(bt, ck['dv_std'])
        pos, cd = m.sample(f, ck['dv_std'], S=S)
        out[b] = pos[0].cpu().numpy()                             # [S,22,F,2]
        roll[b] = m.intention_rollout(f, cd)[0].cpu().numpy()     # [22,F,2]
    return out, roll, ck['cond']


def draw_pitch(ax, x0, x1, y0, y1):
    ax.set_facecolor('#FFFFFF')
    kw = dict(fill=False, ec=LINE, lw=1.6)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, **kw))
    ax.plot([0, 0], [-34, 34], color=LINE, lw=1.6)
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for s in (1, -1):
        ax.add_patch(Rectangle((s * 52.5 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((s * 52.5 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, **kw))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default='DFL-MAT-J03WMX')
    ap.add_argument('--frame', type=int, default=43680)
    ap.add_argument('--run', default='bdist_s0')
    ap.add_argument('--S', type=int, default=20)
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--runs_dir', default='./data/dfl/intent_gen/runs')
    ap.add_argument('--out_dir', default='./data/dfl/intent_gen/results/anim')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--half_width', type=float, default=22.0, help='view half-width in m around the passer')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for fp in ('/root/.fonts/NotoSansCJKkr-Regular.otf',):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = 'Noto Sans CJK KR'

    sc = load_scene(args.sample_dir, args.game, args.frame)
    gen, roll, cond = iad_samples(sc, os.path.join(args.runs_dir, args.run), args.S)
    pos, ball, mask = sc['pos'], sc['ball'], sc['mask']       # [22,76,2], [76,2]
    steps = (np.arange(1, F + 1) * DT)[None, :, None]
    cv = pos[:, H:H + 1] + sc['vel_pausa'][:, None] * steps         # PAUSA HEAD formula
    panels = [('실제', pos[:, H + 1:], None, 'real'),
              ('PAUSA 등속 · 패스 X', cv, None, 'carrier'),
              ('의도만 적분 · b = 패스 X', roll[0], None, 'carrier'),
              ('IAD · b = 패스 O', gen[1].mean(0), gen[1], 'real_hollow'),
              ('IAD · b = 패스 X', gen[0].mean(0), gen[0], 'carrier'),
              ('의도만 적분 · b = 패스 O', roll[1], None, 'real_hollow')]

    # view box: centred on the passer at t
    cx, cy = pos[0, H]
    half_w, half_h = args.half_width, args.half_width * 0.78
    # keep the window inside the pitch plus a 3 m margin, so touchline scenes do not waste half the panel
    cx = min(max(cx, -55.5 + half_w), 55.5 - half_w)
    cy = min(max(cy, -37 + half_h), 37 - half_h)
    x0, x1, y0, y1 = cx - half_w, cx + half_w, cy - half_h, cy + half_h

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.6))
    fig.patch.set_facecolor('#FFFFFF')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.07, wspace=0.03, hspace=0.12)
    art = []
    cols = np.where(sc['team'] == 1, RED, BLUE)
    for ax, (title, fut, samples, ball_mode) in zip(axes.ravel(), panels):
        draw_pitch(ax, x0, x1, y0, y1)
        ax.set_title(title, fontsize=15, color=INK, loc='left', pad=6)
        trails = [ax.plot([], [], color=cols[j], lw=2.0, alpha=0.45, solid_capstyle='round')[0] for j in range(22)]
        # thin lines = each IAD sample's path since t
        fan = [ax.plot([], [], color=cols[j % 22], lw=0.7, alpha=0.22)[0]
               for j in range(22 * (args.S if samples is not None else 0))]
        ghost = ax.scatter(np.zeros(22), np.zeros(22), s=150, facecolors='none', edgecolors='#9CA3AF',
                           linewidths=1.3, linestyle='--', zorder=3)
        dots = ax.scatter(np.zeros(22), np.zeros(22), s=150, c=cols, edgecolors='white', linewidths=1.2, zorder=5)
        ring = ax.scatter([0], [0], s=330, facecolors='none', edgecolors=INK, linewidths=1.8, zorder=6)
        bsolid = ax.scatter([0], [0], s=55, c=INK, zorder=7)
        bhollow = ax.scatter([0], [0], s=70, facecolors='white', edgecolors=INK, linewidths=1.6, zorder=7)
        art.append(dict(trails=trails, fan=fan, ghost=ghost, dots=dots, ring=ring, bsolid=bsolid, bhollow=bhollow,
                        fut=fut, samples=samples, ball_mode=ball_mode))
    stamp = fig.text(0.5, 0.945, '', ha='center', va='center', fontsize=18, color=INK)
    fig.text(0.5, 0.025, '● 공   ○ 공은 생성하지 않음(패스 X 판은 passer 위치, 패스 O 판은 실제 공 위치)   ◎ passer   '
             '점선 원 = 같은 시각의 실제 위치   가는 선 = IAD 표본 20개의 경로   의도만 적분 = 예측한 가속도(0.2초×5)를 적분, 디퓨전 없음',
             ha='center', fontsize=12, color='#6B7280')

    idx_seq = list(range(H - PAST_SHOW, H)) + [H] * (HOLD_T + 1) + list(range(H + 1, H + F + 1)) + [H + F] * HOLD_END

    def frame_pos(a, k):
        """position of every player at absolute frame index k (0..H+F)."""
        if k <= H:
            return pos[:, k]
        return a['fut'][:, k - H - 1]

    def update(n):
        k = idx_seq[n]
        t = (k - H) * DT
        stamp.set_text(f't = {t:+.2f} s   ' + ('관측' if k <= H else '생성 구간'))
        for a in art:
            cur = frame_pos(a, k)
            a['dots'].set_offsets(np.where(mask[:, None], cur, np.nan))
            a['ring'].set_offsets(cur[:1])
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
                b = ball[k] if a['ball_mode'] == 'real_hollow' else cur[0]
                a['bhollow'].set_offsets(b[None])
        return []

    tag = f'{args.game}_{args.frame}_{args.run}'
    for k_snap, name in ((H, 't0'), (H + 12, 't+0.5'), (H + F, 't+1.0')):
        update(idx_seq.index(k_snap) if k_snap != H else PAST_SHOW)
        fig.savefig(os.path.join(args.out_dir, f'{tag}_{name}.png'), dpi=110)
    ani = animation.FuncAnimation(fig, update, frames=len(idx_seq), blit=False)
    gif = os.path.join(args.out_dir, f'{tag}.gif')
    ani.save(gif, writer=animation.PillowWriter(fps=args.fps), dpi=80)
    # H.264 mp4 via the static ffmpeg from imageio-ffmpeg (system ffmpeg's openh264 is broken; mpeg4 won't play in browsers)
    ffs = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 '.venv/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*'))
    if ffs:
        subprocess.run([ffs[0], '-loglevel', 'error', '-y', '-i', gif, '-movflags', 'faststart', '-pix_fmt', 'yuv420p',
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-crf', '20',
                        os.path.join(args.out_dir, f'{tag}.mp4')], check=False)
    else:
        print('mp4 skipped: imageio-ffmpeg not installed in .venv')

    # short numbers: defenders within 10 m of the passer, distance change over 1 s
    p_t = pos[:, H]
    dist = np.linalg.norm(p_t - p_t[:1], axis=-1)
    near = mask & (sc['team'] == 0) & (dist < 10)
    print('cond', cond, '| defenders <10 m:', int(near.sum()), '| codes', sc['codes'][near].tolist())
    for title, fut, samples, _ in panels:
        end = fut[:, -1]
        dd = np.linalg.norm(end - end[:1], axis=-1) - dist
        err = np.linalg.norm(fut - pos[:, H + 1:], axis=-1).mean(-1)
        msg = f'{title:22s} 1초 뒤 passer까지 거리 변화(m): {np.round(dd[near], 2).tolist()} | 1초 평균 오차(m) 전체 {err[mask].mean():.2f}, 근처 수비 {np.round(err[near], 2).tolist()}'
        if samples is not None:
            sd = samples[:, :, -1].std(0).mean(-1)      # per player, spread of the 20 end positions
            msg += f' | 표본 끝점 퍼짐(m) 전체 {sd[mask].mean():.2f}, passer {sd[0]:.2f}, 근처 수비 {np.round(sd[near], 2).tolist()}'
        print(msg)
    print('saved', os.path.join(args.out_dir, tag + '.gif'))


if __name__ == '__main__':
    main()
