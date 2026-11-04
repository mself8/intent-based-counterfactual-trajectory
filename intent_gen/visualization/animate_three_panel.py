"""3판 영상 — 패스한 장면의 반사실(no-pass) 궤적을 PAUSA와 Ours가 각각 어떻게 생성하는지 비교한다.

판 3개: 실제 궤적(패스 수행) · PAUSA 반사실 궤적(등속 가정) · Ours 반사실 궤적(적분 궤적, b=0).
Ours는 PAUSA에 넣을 적분 궤적 하나만 그린다(디퓨전 표본은 넣지 않음).
반사실 세계는 드리블이 전제라 PAUSA·Ours 판 모두 공을 passer 발밑에 그린다
(PAUSA 코드도 가상 공을 직선으로 늘리지만 점수 계산엔 쓰지 않고 passer 위치를 공 위치로 쓴다). 결과는 mp4만 남긴다.

장면 = 시험 경기(보훔–레버쿠젠)의 패스 장면 중 경기 시간에 고르게 퍼진 100개(--n). 원터치 패스(t-0.4초에 공이 3 m 넘게 떨어져 있다가
초속 8 m 넘게 날아온 장면)는 관측 구간에 들어오는 패스가 보여 헷갈려서 뺀다.
파일 이름 = 경기_전반·후반_영상이 시작하는 경기 시각(mm분ss초) → 이름순 = 경기 시간순.

    nice -n 19 /opt/conda/bin/python intent_gen/visualization/animate_three_panel.py --run exp4_all_s0 --shard 0 --n_shards 6
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # intent_gen/
from models.iad import DT, F, H  # noqa: E402
from models.ours import Backbone, split_batch  # noqa: E402
from visualization.animate_exp import TEST_GAME, render, tensors  # noqa: E402
from visualization.animate_scene import PAST_SHOW  # noqa: E402

HOME, AWAY = '보훔', '레버쿠젠'
LEGEND = '● 실제 볼   ○ 반사실 볼 (passer 보유)   ◎ passer   점선 원 = 동일 시점의 실제 위치'


def clock(frame, period):
    """DFL 프레임 번호(전반 10000, 후반 100000에서 시작, 25 Hz) -> 경기 시계 초."""
    return (frame - (10000 if period == 1 else 100000)) / 25 + (0 if period == 1 else 2700)


def mmss(s, sep=':'):
    return f'{int(s // 60):02d}{sep}{int(s % 60):02d}' if sep == ':' else f'{int(s // 60):02d}분{int(s % 60):02d}초'


def scenes(d, n=100):
    """패스 장면(원터치 제외) 중 경기 시간에 고르게 퍼진 n개의 행 번호와 파일 이름, 경기 시간순."""
    d_bp = np.linalg.norm(d['ball'] - d['pos'][:, 0], axis=-1)
    spd = np.linalg.norm(np.diff(d['ball'], axis=1), axis=-1) / DT
    onetouch = (d_bp[:, H - 10] >= 3.0) & (spd[:, H - 15:H].max(1) >= 8.0)
    rows = np.where((d['b'] == 1) & ~onetouch)[0]
    rows = rows[np.lexsort((d['frame'][rows], d['period'][rows]))]
    T = np.array([(int(d['frame'][i]) - (10000 if d['period'][i] == 1 else 100000)) / 25
                  + (0 if d['period'][i] == 1 else 3000) for i in rows])      # 단조 증가하는 경기 경과 초
    if n and n < len(rows):           # 첫 패스~마지막 패스를 n등분한 시각마다 가장 가까운 장면 하나
        used = set()
        for tg in np.linspace(T[0], T[-1], n):
            used.add(next(k for k in np.argsort(np.abs(T - tg)) if k not in used))
        rows = rows[sorted(used)]
    names, out = set(), []
    for i in rows:
        p = int(d['period'][i])
        start = clock(int(d['frame'][i]) - PAST_SHOW, p)
        name = f'{HOME}-{AWAY}_{"전반" if p == 1 else "후반"}_{mmss(start, "")}'
        k = 2
        while name in names:          # 같은 초에 시작하는 장면이 둘이면 _2
            name = f'{name.split("__")[0]}__{k}'
            k += 1
        names.add(name)
        out.append((int(i), name))
    return out, int(onetouch[d['b'] == 1].sum())


@torch.no_grad()
def ours_no_pass(sc, G):
    sb = split_batch(tensors(sc, 0, 'cpu'))
    return G.rollout(sb['P0'], sb['BL0'], sb['meta'])[0][0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='expD_lean_s0')
    ap.add_argument('--out_dir', default='./data/dfl/intent_gen/results/06_pausa_vs_ours')
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--runs_dir', default='./data/dfl/intent_gen/runs')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--n', type=int, default=100, help='만들 장면 수(0 = 전부)')
    args = ap.parse_args()
    torch.set_num_threads(2)
    for fp in ('/root/.fonts/NotoSansCJKkr-Regular.otf',):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = 'Noto Sans CJK KR'

    g = torch.load(os.path.join(args.runs_dir, args.run, 'g.pt'), map_location='cpu')
    G = Backbone(g['feat'], g['ar'])
    G.load_state_dict(g['model'])
    G.eval()

    d = dict(np.load(os.path.join(args.sample_dir, f'{TEST_GAME}.npz')))
    todo, n_onetouch = scenes(d, args.n)
    if args.shard == 0:
        print(f'패스 장면 {len(todo)}개 (원터치 {n_onetouch}개 뺌)', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    ts = (np.arange(1, F + 1) * DT)[:, None]
    for i, name in todo[args.shard::args.n_shards]:
        sc = {k: v[i] for k, v in d.items()}
        cv = sc['pos'][:, H:H + 1] + sc['vel_pausa'][:, None] * ts[None]
        p = int(sc['period'])
        team = HOME if str(sc['codes'][0]).startswith('Home') else AWAY
        start, at = clock(int(sc['frame']) - PAST_SHOW, p), clock(int(sc['frame']) + 1, p)
        panels = [('실제 궤적 (패스 수행)', sc['pos'][:, H + 1:], None, 'real'),
                  ('PAUSA 반사실 궤적 (등속 가정)', cv, None, 'carrier'),
                  ('Ours 반사실 궤적', ours_no_pass(sc, G), None, 'carrier')]
        render(sc, panels, name, args.out_dir, f'{HOME}–{AWAY} {mmss(start)} 시작',
               f'{mmss(at)} {team} 패스', legend=LEGEND)
        if os.path.exists(os.path.join(args.out_dir, name + '.mp4')):   # 영상만 남김
            for ext in ('.gif', '_t0.png', '_t+0.5.png', '_t+1.0.png'):
                os.remove(os.path.join(args.out_dir, name + ext))
        print(name, flush=True)


if __name__ == '__main__':
    main()
