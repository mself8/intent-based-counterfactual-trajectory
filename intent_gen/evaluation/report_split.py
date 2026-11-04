"""장면 종류(공 지킨·패스한) x passer 5 m 안·밖 수비수의 ADE·FDE, 5 m 안 수비수가 좁힌 간격, 시점별 위치 오차.

좁힌 간격 = t의 수비수–passer 거리 - 그 시각 뒤의 수비수–passer 거리(양수 = 가까워짐). 각 세계의 passer 위치로 잰다.
PAUSA 직선은 패스 안 한 세계의 가정이라 패스한 장면에서는 재지 않는다(그 세계의 정답이 없고, PAUSA는 패스한 세계를 만들지 않는다).

    /opt/conda/bin/python intent_gen/evaluation/report_split.py
    FUT_FRAMES=50 /opt/conda/bin/python intent_gen/evaluation/report_split.py --sample_dir ./data/dfl/intent_gen/samples_2s \
        --runs Ours:exp4_2s_s0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # intent_gen/
from evaluation import harness  # noqa: E402
from evaluation.harness import DT, F, H  # noqa: E402

RADIUS = 5.0


def cell(model, scene, text):
    return '–' if model == 'PAUSA' and scene == '패스한 장면' else text


def table(head, rows):
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for r in rows:
        print('| ' + ' | '.join(r) + ' |')
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--runs_dir', default='./data/dfl/intent_gen/runs')
    ap.add_argument('--runs', nargs='*', default=['Ours:expD_lean_s0'])
    args = ap.parse_args()
    d = harness.load_split(args.sample_dir, 'test')
    real = d['pos'][:, :, H + 1:]
    preds = {'PAUSA': harness.cv_pausa(d)[:, 0]}
    for r in args.runs:
        label, run = r.split(':')
        preds[label] = np.load(os.path.join(args.runs_dir, run, 'pred_bb_test.npy'))[:, 0]

    p_t = d['pos'][:, :, H]
    dist = np.linalg.norm(p_t - p_t[:, :1], axis=-1)
    defender = d['mask'] & (d['team'] == 0)
    groups = {'전체 선수': d['mask'], f'{RADIUS:.0f} m 안 수비수': defender & (dist < RADIUS),
              f'{RADIUS:.0f} m 밖 수비수': defender & (dist >= RADIUS)}
    classes = {'공 지킨 장면': d['b'] == 0, '패스한 장면': d['b'] == 1}
    print('장면 수', {k: int(v.sum()) for k, v in classes.items()})
    print('5 m 안 수비수가 있는 장면 비율', {k: round(float((defender & (dist < RADIUS))[v].any(1).mean()), 3)
                                  for k, v in classes.items()}, '\n')

    for sec in [s for s in (1.0, 2.0) if round(s / DT) <= F]:
        f = round(sec / DT)
        err = {k: np.linalg.norm(v[:, :, :f] - real[:, :, :f], axis=-1) for k, v in preds.items()}
        print(f'### {sec:.0f}초 ADE / FDE (cm)\n')
        rows = []
        for cn, c in classes.items():
            for gn, g in groups.items():
                m = g & c[:, None]
                rows.append([cn, gn] + [cell(k, cn, f'{e.mean(-1)[m].mean() * 100:.1f} / {e[..., -1][m].mean() * 100:.1f}')
                                        for k, e in err.items()])
        table(['장면', '선수'] + list(preds), rows)

        print(f'### 5 m 안 수비수가 {sec:.0f}초 동안 좁힌 간격 (m, 양수 = 가까워짐)\n')
        gap = lambda p: dist - np.linalg.norm(p[:, :, f - 1] - p[:, :1, f - 1], axis=-1)  # noqa: E731
        near = groups[f'{RADIUS:.0f} m 안 수비수']
        rows = []
        for cn, c in classes.items():
            m = near & c[:, None]
            rows.append([cn, f'{int(m.sum()):,}', f'{gap(real)[m].mean():.2f}'] +
                        [cell(k, cn, f'{gap(p)[m].mean():.2f}') for k, p in preds.items()])
        table(['', '수', '실제'] + list(preds), rows)
        print(f'### 같은 값의 수비수별 오차 |예측 - 실제| 평균 (m)\n')
        table(['', '수'] + list(preds),
              [[cn, f'{int((near & c[:, None]).sum()):,}'] +
               [cell(k, cn, f'{np.abs(gap(p) - gap(real))[near & c[:, None]].mean():.2f}') for k, p in preds.items()]
               for cn, c in classes.items()])

    keep = classes['공 지킨 장면']
    print('### 시점별 위치 오차, 공 지킨 장면 (cm)\n')
    passer = d['mask'] & (np.arange(22)[None] == 0)
    for gn, g in {'전체 선수': groups['전체 선수'], 'passer': passer, '5 m 안 수비수': groups[f'{RADIUS:.0f} m 안 수비수']}.items():
        rows = []
        for f in range(5, F + 1, 5):
            m = g & keep[:, None]
            rows.append([f'{f * DT:.1f}초'] + [f'{np.linalg.norm(v[:, :, f - 1] - real[:, :, f - 1], axis=-1)[m].mean() * 100:.1f}'
                                              for v in preds.values()])
        print(gn)
        table(['시점'] + list(preds), rows)


if __name__ == '__main__':
    main()
