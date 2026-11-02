"""1차 라운드 결과 집계: 기준선 + 학습 모델의 시험 경기 지표를 표로.

    python intent_gen/report.py --runs past_s0 bdist_s0
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

GROUPS = ['all', 'carrier', 'def_0-5m', 'def_5-10m', 'def_10-20m', 'def_20-infm', 'teammates']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['past_s0', 'bdist_s0'])
    ap.add_argument('--runs_dir', default='./data/dfl/intent_gen/runs')
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--out', default='./data/dfl/intent_gen/results/round1.md')
    args = ap.parse_args()

    d = harness.load_split(args.sample_dir, 'test')
    dfs = [harness.evaluate(fn(d), d, name) for name, fn in
           [('cv_pausa', harness.cv_pausa), ('cv_past', harness.cv_past)]]
    phys = [harness.physics(fn(d), d, name) for name, fn in
            [('cv_pausa', harness.cv_pausa), ('cv_past', harness.cv_past)]]
    best = {}
    for r in args.runs:
        dfs.append(pd.read_csv(os.path.join(args.runs_dir, r, 'test_metrics.csv')))
        with open(os.path.join(args.runs_dir, r, 'log.json')) as fh:
            lg = json.load(fh)
        phys.append(lg['physics'])
        best[r] = (lg['best_epoch'], min(e['val_ADE'] for e in lg['epochs'] if 'val_ADE' in e))
    df = pd.concat(dfs)

    lines = ['# 1차 라운드 — 시험 경기(보훔–레버쿠젠) 1초 생성 오차', '']
    for metric in ('ADE', 'FDE', 'minADE', 'minFDE'):
        t = df[(df.metric == metric) & df.group.isin(GROUPS)].pivot_table(
            index=['class', 'group'], columns='model', values='value')
        t = t.reindex(columns=[c for c in ['cv_pausa', 'cv_past'] + args.runs if c in t.columns])
        lines += [f'## {metric} (m)', '', t.round(3).to_string(), '']
    p = df[df.group == 'press_def<10m'].pivot_table(index=['class', 'metric'], columns='model', values='value')
    lines += ['## 소유자 10 m 안 수비수의 1초 거리 변화 (m, 음수 = 다가감)', '', p.round(3).to_string(), '']
    lines += ['## 물리 위반 비율', '', pd.DataFrame(phys).round(4).to_string(index=False), '']
    lines += ['## 체크포인트', '', json.dumps(best), '']
    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(text)
    print(text)


if __name__ == '__main__':
    main()
