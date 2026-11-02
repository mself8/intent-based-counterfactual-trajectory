"""반사실 1초 생성기 검증 하네스.

정답이 있는 두 종류 장면에서 생성한 1초를 실제 1초와 비교한다.
  패스 X (b=0): 소유자가 실제로 공을 1초 더 가진 장면  -> PAUSA 반사실과 같은 상황
  패스 O (b=1): 실제로 패스한 장면
생성기 입력은 build_dataset.py 표본(과거 2초, t 포함), 출력은 [N, S, 22, 25, 2] 미래 위치(m).

    python intent_gen/harness.py --split test
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

GAMES = ['DFL-MAT-J03WMX', 'DFL-MAT-J03WN1', 'DFL-MAT-J03WOH', 'DFL-MAT-J03WOY',
         'DFL-MAT-J03WPY', 'DFL-MAT-J03WQQ', 'DFL-MAT-J03WR9']
SPLITS = {'test': ['DFL-MAT-J03WN1'], 'val': ['DFL-MAT-J03WR9']}
SPLITS['train'] = [g for g in GAMES if g not in SPLITS['test'] + SPLITS['val']]
H, F, DT = 50, 25, 0.04
DEF_BINS = [(0, 5), (5, 10), (10, 20), (20, np.inf)]
PRESS_RADIUS = 10.0


def load_split(sample_dir, split):
    parts = [dict(np.load(os.path.join(sample_dir, f'{g}.npz'))) | {'game': None} for g in SPLITS[split]]
    for g, p in zip(SPLITS[split], parts):
        p['game'] = np.array([g] * len(p['b']))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def future_steps():
    return (np.arange(1, F + 1) * DT)[None, None, None, :, None]  # [1,1,1,F,1]


def static(d):
    return np.repeat(d['pos'][:, None, :, H:H + 1], F, axis=3)


def cv_past(d, lag=5):
    """과거만 쓰는 등속: 마지막 0.2초 평균 속도."""
    p0 = d['pos'][:, None, :, H:H + 1]
    v = (d['pos'][:, :, H] - d['pos'][:, :, H - lag]) / (lag * DT)
    return p0 + v[:, None, :, None] * future_steps()


def cv_pausa(d):
    """PAUSA 원판 등속(obso.generate_virtual_trajectory): t 위치 + t 속도(savgol, 미래 3프레임 포함) x 시간."""
    p0 = d['pos'][:, None, :, H:H + 1]
    return p0 + d['vel_pausa'][:, None, :, None] * future_steps()


def group_masks(d):
    """선수 묶음: 소유자 / 같은 팀 / 수비수(소유자와의 거리 구간별)."""
    p_t = d['pos'][:, :, H]
    dist = np.linalg.norm(p_t - p_t[:, :1], axis=-1)  # carrier is slot 0
    att = d['team'] == 1
    valid = d['mask']
    g = {'all': valid,
         'carrier': valid & (np.arange(22)[None] == 0),
         'teammates': valid & att & (np.arange(22)[None] > 0),
         'defenders': valid & ~att}
    for lo, hi in DEF_BINS:
        g[f'def_{lo}-{hi if np.isfinite(hi) else "inf"}m'] = valid & ~att & (dist >= lo) & (dist < hi)
    return g, dist


def evaluate(pred, d, name):
    """pred [N,S,22,F,2] -> long-form rows (class x group x metric)."""
    gt = d['pos'][:, None, :, H + 1:]
    err = np.linalg.norm(pred - gt, axis=-1)                    # [N,S,22,F]
    ade_s, fde_s = err.mean(-1), err[..., -1]                   # [N,S,22]
    mean_pred = pred.mean(1, keepdims=True)
    err_mean = np.linalg.norm(mean_pred - gt, axis=-1)[:, 0]   # [N,22,F]
    per = {'minADE': ade_s.min(1), 'minFDE': fde_s.min(1),
           'ADE': ade_s.mean(1), 'FDE': fde_s.mean(1),
           'ADE_of_mean': err_mean.mean(-1), 'FDE_of_mean': err_mean[..., -1]}
    groups, dist = group_masks(d)

    # pressing response: change of distance to the carrier over 1 s (defenders within PRESS_RADIUS at t)
    car_gt, car_pred = gt[:, :, :1, -1], pred[:, :, :1, -1]
    dd_gt = np.linalg.norm(gt[:, :, :, -1] - car_gt, axis=-1)[:, 0] - dist
    dd_pred = (np.linalg.norm(pred[:, :, :, -1] - car_pred, axis=-1) - dist[:, None]).mean(1)
    press = d['mask'] & (d['team'] == 0) & (dist < PRESS_RADIUS) & (dist > 0)

    rows = []
    for cls_name, cls in [('패스 X', d['b'] == 0), ('패스 O', d['b'] == 1)]:
        for gname, gm in groups.items():
            m = gm & cls[:, None]
            if not m.any():
                continue
            for k, v in per.items():
                rows.append((name, cls_name, gname, k, float(v[m].mean()), int(m.sum())))
        pm = press & cls[:, None]
        rows.append((name, cls_name, 'press_def<10m', 'dDist_gt', float(dd_gt[pm].mean()), int(pm.sum())))
        rows.append((name, cls_name, 'press_def<10m', 'dDist_pred', float(dd_pred[pm].mean()), int(pm.sum())))
        rows.append((name, cls_name, 'press_def<10m', 'dDist_absErr',
                     float(np.abs(dd_pred - dd_gt)[pm].mean()), int(pm.sum())))
    return pd.DataFrame(rows, columns=['model', 'class', 'group', 'metric', 'value', 'n'])


def physics(pred, d, name):
    """속도·가속도 상한 위반 비율(생성 1초 구간)."""
    full = np.concatenate([np.repeat(d['pos'][:, None, :, H - 1:H + 1], pred.shape[1], 1), pred], axis=3)
    v = np.diff(full, axis=3) / DT
    a = np.diff(v, axis=3) / DT
    m = d['mask'][:, None, :, None]
    sp = np.linalg.norm(v[:, :, :, 1:], axis=-1)
    ac = np.linalg.norm(a, axis=-1)
    return {'model': name, 'speed>10m/s': float((sp > 10)[np.broadcast_to(m, sp.shape)].mean()),
            'accel>10m/s2': float((ac > 10)[np.broadcast_to(m, ac.shape)].mean())}


def summary_table(df, metrics=('ADE', 'FDE', 'minADE', 'minFDE')):
    t = df[df.metric.isin(metrics)].pivot_table(index=['class', 'group'], columns=['model', 'metric'], values='value')
    return t.round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out_dir', default='./data/dfl/intent_gen/results')
    args = ap.parse_args()
    d = load_split(args.sample_dir, args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    dfs, phys = [], []
    for name, fn in [('static', static), ('cv_past', cv_past), ('cv_pausa', cv_pausa)]:
        pred = fn(d)
        dfs.append(evaluate(pred, d, name))
        phys.append(physics(pred, d, name))
    df = pd.concat(dfs)
    df.to_csv(os.path.join(args.out_dir, f'baselines_{args.split}.csv'), index=False)
    pd.set_option('display.width', 250)
    print(summary_table(df, ('ADE', 'FDE')))
    print(df[df.group == 'press_def<10m'].pivot_table(index=['class', 'metric'], columns='model', values='value').round(3))
    print(json.dumps(phys, ensure_ascii=False))


if __name__ == '__main__':
    main()
