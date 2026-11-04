"""2차 라운드 집계: 결과 폴더 맨 위 README(실험 목록·시험 경기 숫자)와 실험 폴더별 README(장면별 숫자).

    CUDA_VISIBLE_DEVICES=3 nice -n 19 /opt/conda/bin/python intent_gen/evaluation/report_ours.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # intent_gen/
from evaluation import harness  # noqa: E402
from models.iad import DT, H, IADSoccer, prepare  # noqa: E402
from training.train_iad import to_tensors  # noqa: E402
from training.train_ours import b_effect  # noqa: E402

EXPS = [  # run, folder, label, 무엇을 바꿨나
    ('exp1_bb_s0', '02_exp1_backbone_residual', '실험1', '의도 적분 궤적 + 잔차 디퓨전',
     ['의도 적분 궤적을 기본으로 쓰고, 디퓨전은 \'실제 − 적분 궤적\' 잔차만 만든다(0.2초 마디 5개, 마디 사이는 직선).',
      '입력은 1차와 같다: 과거 2초, 공 상태 b, passer와의 거리.',
      '1차의 도착점 모듈은 뺐다. 적분 궤적은 의도 손실(가속 방향·크기)에 1초 위치 오차를 더해 학습했다.']),
    ('exp2_rel_s0', '03_exp2_relative_weighted', '실험2', '실험1 + 상대 좌표·φ·재껴짐 + 가중',
     ['실험1에 입력 추가: passer 기준 상대 좌표·상대 속도, 과거 가속도의 passer 방향 성분(φ), 재껴짐 여부, 접근 속도.',
      '학습 때 passer 10 m 안 수비수는 3배, 큰 가속은 최대 3배 가중.']),
    ('exp3_ar_s0', '04_exp3_autoregressive', '실험3', '실험1 + 0.2초 자기회귀',
     ['실험1을 자기회귀로: 0.2초 의도 하나를 예측해 움직이고, 바뀐 장면을 다시 읽어 다음 0.2초를 예측한다(5번).',
      '학습도 자기 예측으로 굴려서 한다. t 뒤의 공은 b = 패스 X면 passer 위치, 패스 O면 t 위치에 둔다.']),
    ('exp4_all_s0', '05_exp4_all', '실험4', '실험2 + 실험3',
     ['실험2의 입력·가중과 실험3의 자기회귀를 함께 쓴다.']),
]


def pick(df, model, cls, group, metric):
    r = df[(df.model == model) & (df['class'] == cls) & (df.group == group) & (df.metric == metric)]
    return float(r.value.iloc[0]) if len(r) else float('nan')


def main_row(df, model, beff=None):
    return {'ADE 패스 X': pick(df, model, '패스 X', 'all', 'ADE'),
            '5 m 안 수비수 ADE': pick(df, model, '패스 X', 'def_0-5m', 'ADE'),
            '압박 패스 X': pick(df, model, '패스 X', 'press_def<10m', 'dDist_pred'),
            '압박 패스 O': pick(df, model, '패스 O', 'press_def<10m', 'dDist_pred'),
            'b 효과 0–5 m': beff['0-5m'] if beff else float('nan'),
            'b 효과 20 m+': beff['20-infm'] if beff else float('nan')}


def md_table(rows, cols, fmt='{:.3f}'):
    lines = ['| | ' + ' | '.join(cols) + ' |', '|---|' + '---|' * len(cols)]
    for name, r in rows:
        cells = ['' if (isinstance(r.get(c), float) and np.isnan(r[c])) or c not in r
                 else (fmt.format(r[c]) if isinstance(r[c], float) else str(r[c])) for c in cols]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _hist_feats(d):
    """t까지 프레임만 쓴 장면 요약: 공 속도·공–passer 거리·passer 속도·가속·위치·가장 가까운 수비수."""
    pos, ball, team, mask = d['pos'], d['ball'], d['team'], d['mask']
    car = pos[:, 0]
    dist = np.linalg.norm(pos[:, :, H] - car[:, None, H], axis=-1)
    dprev = np.linalg.norm(pos[:, :, H - 5] - car[:, None, H - 5], axis=-1)
    dm = np.where(mask & (team == 0), dist, np.inf)
    j = dm.argmin(1)
    n = np.arange(len(j))
    return np.stack([np.linalg.norm(ball[:, H] - ball[:, H - 1], axis=-1) / DT,
                     np.linalg.norm(ball[:, H] - ball[:, H - 5], axis=-1) / 0.2,
                     np.linalg.norm(ball[:, H] - car[:, H], axis=-1),
                     np.linalg.norm(ball[:, H - 5] - car[:, H - 5], axis=-1),
                     np.linalg.norm(car[:, H] - car[:, H - 3], axis=-1) / (3 * DT),
                     np.linalg.norm((car[:, H] - car[:, H - 5]) - (car[:, H - 5] - car[:, H - 10]), axis=-1) / 0.04,
                     car[:, H, 0], np.abs(car[:, H, 1]), dm[n, j], (dprev[n, j] - dist[n, j]) / 0.2,
                     (dm < 5).sum(1), (dm < 10).sum(1)], 1)


def pass_like(sample_dir, d, q=0.85):
    """패스할 것처럼 보였는데 공을 지킨 장면 = 학습 경기로 '과거만 보고 패스 여부' 분류기를 학습해,
    시험 경기 패스 X 장면 중 패스 확률 상위 15%. '패스 직전 같은 과거 + 패스 X' 반사실의 정답이 있는 유일한 장면."""
    tr = harness.load_split(sample_dir, 'train')
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    clf.fit(_hist_feats(tr), tr['b'])
    p = clf.predict_proba(_hist_feats(d))[:, 1]
    X = d['b'] == 0
    return X & (p >= np.quantile(p[X], q))


def press_sel(pred, d, sel):
    """passer 10 m 안 수비수의 1초 거리 변화(m), 표본 평균, 장면 sel만."""
    p_t = d['pos'][:, :, H]
    dist = np.linalg.norm(p_t - p_t[:, :1], axis=-1)
    dd = (np.linalg.norm(pred[:, :, :, -1] - pred[:, :, :1, -1], axis=-1) - dist[:, None]).mean(1)
    m = d['mask'] & (d['team'] == 0) & (dist < 10) & (dist > 0) & sel[:, None]
    return float(dd[m].mean())


@torch.no_grad()
def round1_rollout(d, run_dir, dev, bs=256):
    """1차 의도 적분: bdist_s0 의도 예측기 출력만 적분, 실제 b와 뒤집은 b."""
    ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location=dev)
    m = IADSoccer(ck['cond']).to(dev)
    m.load_state_dict(ck['model'])
    m.eval()
    T = to_tensors(d, dev)
    out = {}
    for swap in (False, True):
        parts = []
        for i in range(0, len(T['b']), bs):
            bt = {k: v[i:i + bs] for k, v in T.items()}
            if swap:
                bt['b'] = 1 - bt['b']
            parts.append(m.intention_rollout(prepare(bt, ck['dv_std'])).cpu().numpy())
        out[swap] = np.concatenate(parts)[:, None]
    return out[False], out[True]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--runs_dir', default='./data/dfl/intent_gen/runs')
    ap.add_argument('--results', default='./data/dfl/intent_gen/results')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = harness.load_split(args.sample_dir, 'test')

    # 기준선 · 1차
    dfs = [harness.evaluate(harness.cv_pausa(d), d, 'cv_pausa'),
           pd.read_csv(os.path.join(args.runs_dir, 'bdist_s0', 'test_metrics.csv'))]
    iad_s = np.load(os.path.join(args.runs_dir, 'bdist_s0', 'pred_test.npy'))
    iad = iad_s.mean(1, keepdims=True)
    iad_sw = np.load(os.path.join(args.runs_dir, 'bdist_s0', 'pred_test_bswap.npy')).mean(1, keepdims=True)
    r1, r1_sw = round1_rollout(d, os.path.join(args.runs_dir, 'bdist_s0'), dev)
    dfs.append(harness.evaluate(r1, d, 'round1_roll'))
    df = pd.concat(dfs + [pd.read_csv(os.path.join(args.runs_dir, r, 'test_metrics.csv'))
                          for r, *_ in EXPS if os.path.exists(os.path.join(args.runs_dir, r, 'test_metrics.csv'))])
    real = {'ADE 패스 X': float('nan'), '5 m 안 수비수 ADE': float('nan'),
            '압박 패스 X': pick(df, 'cv_pausa', '패스 X', 'press_def<10m', 'dDist_gt'),
            '압박 패스 O': pick(df, 'cv_pausa', '패스 O', 'press_def<10m', 'dDist_gt')}
    rows = [('실제', real), ('PAUSA 등속', main_row(df, 'cv_pausa')),
            ('1차 IAD', main_row(df, 'bdist_s0', b_effect(iad, iad_sw, d))),
            ('1차 의도 적분', main_row(df, 'round1_roll', b_effect(r1, r1_sw, d)))]
    looks = pass_like(args.sample_dir, d)
    for (_, r), pred in zip(rows, [d['pos'][:, None, :, H + 1:], harness.cv_pausa(d), iad_s, r1]):
        r['압박 패스 같던 X'] = press_sel(pred, d, looks)
    srows = [('1차 IAD', {'ADE': pick(df, 'bdist_s0', '패스 X', 'all', 'ADE'),
                         'minADE': pick(df, 'bdist_s0', '패스 X', 'all', 'minADE'),
                         '표본 평균의 ADE': pick(df, 'bdist_s0', '패스 X', 'all', 'ADE_of_mean')})]
    logs = {}
    for run, folder, label, desc, what in EXPS:
        lp = os.path.join(args.runs_dir, run, 'log.json')
        if not os.path.exists(lp):
            continue
        with open(lp) as fh:
            logs[run] = json.load(fh)
        rows.append((f'{label} 적분 궤적', main_row(df, run + '_bb', logs[run]['b_effect_bb'])))
        rows[-1][1]['압박 패스 같던 X'] = press_sel(np.load(os.path.join(args.runs_dir, run, 'pred_bb_test.npy')), d, looks)
        srows.append((label, {'ADE': pick(df, run, '패스 X', 'all', 'ADE'),
                              'minADE': pick(df, run, '패스 X', 'all', 'minADE'),
                              '표본 평균의 ADE': pick(df, run, '패스 X', 'all', 'ADE_of_mean')}))
    cols = ['ADE 패스 X', '5 m 안 수비수 ADE', '압박 패스 X', '압박 패스 O', '압박 패스 같던 X', 'b 효과 0–5 m', 'b 효과 20 m+']
    main_tab = md_table(rows, cols)
    samp_tab = md_table(srows, ['ADE', 'minADE', '표본 평균의 ADE'])

    folders = [('01_round1_iad', '1차 (10-30~31): IAD 축구판 + 의도 적분 진단. 장면 8개')] + \
        [(f, f'{lab}: {desc}') for r, f, lab, desc, _ in EXPS if r in logs] + \
        [('06_pausa_vs_ours', "3판 영상: 실제(패스함) · PAUSA · 우리 모델(실험4)의 '패스 안 했다면'. 시험 경기 패스 장면 중 경기 시간에 고르게 퍼진 100개(원터치 제외), 두 판 모두 공은 passer가 몰고 감, 파일 이름 = 경기_전후반_시작 시각(이름순 = 시간순), mp4만"),
         ('07_pausa_vs_ours_2s', '06과 같은 3판 영상의 2초 버전: Ours를 미래 2초로 다시 학습(런 exp4_2s_s0, 장면 samples_2s/, 공 지킨 장면은 2초 내내 공이 passer 3 m 안), 패스 장면 100개, mp4만')]
    readme = ['# 반사실 1초 궤적 생성 — 결과', '',
              '실험마다 폴더 하나. 폴더 안 `pass_X/`, `pass_O/`에 장면별 애니메이션이 있다.', '',
              '- `pass_X` = 실제로 공을 1초 더 가진 장면. PAUSA 반사실(패스를 미뤘다면)과 같은 상황이라 정답이 있다.',
              '- `pass_O` = 실제로 패스한 장면.',
              '- 장면마다 `.mp4` 하나. 이름 끝 `_rand` = 무작위 장면(2차부터).',
              '- 판 6개: 위 = 실제 · PAUSA 등속(패스 X) · 1차 의도 적분(패스 X), '
              '아래 = 이 실험 b=패스 O · 이 실험 b=패스 X · 1차 의도 적분(패스 O).',
              '- 2차 실험 판의 굵은 점 = 적분 궤적(PAUSA에 넣을 결정론 궤적), 가는 선 = 잔차 디퓨전 표본 20개.', '',
              '| 폴더 | 내용 |', '|---|---|'] + [f'| `{f}/` | {t} |' for f, t in folders] + [
              '', '실험4 = Ours. 장면·선수별 ADE / FDE와 5 m 안 수비수가 다가간 거리는 `tables_1s.md`, `tables_2s.md`(`intent_gen/evaluation/report_split.py`).',
              '', '## 시험 경기 숫자 (보훔–레버쿠젠, 1초)', '',
              'ADE = 1초 평균 위치 오차(m). 압박 = passer 10 m 안 수비수의 1초 거리 변화(m, 음수 = 다가감, 실제와 가까울수록 좋음). '
              'b 효과 = b를 뒤집었을 때 1초 뒤 위치 차이(m), passer와의 거리 구간별.', '',
              f'\'압박 패스 같던 X\' = 과거만 보면 패스할 것 같았는데(과거만 쓰는 분류기 상위 15%, {int(looks.sum())}장면) 실제로는 공을 지킨 장면. '
              '모델이 b 대신 \'패스 직전 같은 과거\'에 기대면 여기서 실제보다 덜 붙는다(누수·혼입 점검).', '',
              main_tab, '', '### 표본 20개 (패스 X, m)', '', samp_tab, '',
              '씨앗 1개(0). 2차 실험은 적분 궤적 80에폭(검증 ADE 최저 체크포인트) + 잔차 디퓨전 60에폭.', '']
    with open(os.path.join(args.results, 'README.md'), 'w') as fh:
        fh.write('\n'.join(readme))
    print('\n'.join(readme))

    # 실험 폴더별 README
    for run, folder, label, desc, what in EXPS:
        if run not in logs:
            continue
        lg = logs[run]
        sp = os.path.join(args.results, folder, 'scene_stats.json')
        stats = json.load(open(sp)) if os.path.exists(sp) else []
        lines = [f'# {label} · {desc}', '', '## 무엇을 바꿨나', ''] + [f'- {w}' for w in what] + [
            '', f'학습: 적분 궤적 {lg["args"]["epochs_g"]}에폭 중 검증 최고 {lg["g_best_epoch"]}에폭(검증 ADE {lg["g_val_ADE"]:.3f}), '
            f'잔차 디퓨전 {lg["args"]["epochs_d"]}에폭, 씨앗 {lg["args"]["seed"]}, {lg["minutes"]}분.', '',
            '## 시험 경기 숫자 (적분 궤적)', '',
            md_table([rows[0], rows[1], rows[3], (f'{label} 적분 궤적', dict(rows)[f'{label} 적분 궤적'])], cols), '',
            '## 장면별 — 볼 수비수의 1초 거리 변화 (m, 음수 = passer에게 다가감)', '',
            '1차와 같은 장면은 1차 표와 같은 수비수, 무작위 장면은 passer에게 가장 가까운 수비수. '
            '\'이 실험\'은 실제와 같은 b, \'반대 b\'는 b를 뒤집은 반사실.', '',
            '| 파일 | 장면 | 수비수(t 거리) | 실제 | PAUSA 등속 | 1차 의도 적분 | 이 실험 | 이 실험, 반대 b |',
            '|---|---|---|---|---|---|---|---|']
        for kind in ('pass_X', 'pass_O'):
            for s in [x for x in stats if x['kind'] == kind]:
                same, other = ('X', 'O') if kind == 'pass_X' else ('O', 'X')
                lines.append(f"| `{kind}/{s['file']}` | {s['note']} | {s['defender']} ({s['dist_t']} m) | {s['real']:+.2f} | "
                             f"{s['cv']:+.2f} | {s['ref_' + same]:+.2f} | {s['exp_' + same]:+.2f} | {s['exp_' + other]:+.2f} |")
        with open(os.path.join(args.results, folder, 'README.md'), 'w') as fh:
            fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
