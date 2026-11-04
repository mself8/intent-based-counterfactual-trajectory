"""2차 라운드 학습·평가: 1단계 적분 궤적(의도 적분) -> 2단계 잔차 디퓨전(적분 궤적 고정) -> 시험 경기 예측·지표.

    CUDA_VISIBLE_DEVICES=3 nice -n 19 /opt/conda/bin/python -W ignore intent_gen/training/train_ours.py --run exp1_bb_s0 --feat bdist
    ... --run exp2_rel_s0 --feat rel --weighted       (②)
    ... --run exp3_ar_s0  --feat bdist --ar           (③)
    ... --run exp4_all_s0 --feat rel --weighted --ar  (②+③)
    ... --run expA_terrain_s0 --feat rel_terrain --weighted --ar --bb_only  (실험4 + 골대·측면선 거리)
    ... --run expC_bd3_s0 --feat rel_bd3 --weighted --ar --bb_only          (실험4에서 b·거리 입력 3개로 축소)
    ... --run expD_lean_s0 --feat rel_lean --weighted --ar --bb_only        (Ours: C에서 재껴짐까지 제거)
    FUT_FRAMES=50 ... --run exp4_2s_s0 --feat rel --weighted --ar --bb_only --sample_dir ./data/dfl/intent_gen/samples_2s

적분 궤적은 검증 경기 ADE가 가장 낮은 체크포인트를, 잔차 디퓨전은 마지막 가중치를 쓴다.
시험 경기 예측: 적분 궤적(결정론) + 표본 20개, 각각 실제 b와 뒤집은 b.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # intent_gen/
from evaluation import harness  # noqa: E402
from models.ours import H, STEP, Backbone, ResDiff, split_batch  # noqa: E402
from training.train_iad import to_tensors  # noqa: E402


def _slices(n, bs):
    return [slice(i, min(i + bs, n)) for i in range(0, n, bs)]


def _batch(T, sl, swap=False):
    bt = {k: v[sl] for k, v in T.items()}
    if swap:
        bt['b'] = 1 - bt['b']
    return split_batch(bt)


@torch.no_grad()
def predict_bb(G, T, bs=256, swap=False):
    G.eval()
    outs = []
    for sl in _slices(len(T['b']), bs):
        sb = _batch(T, sl, swap)
        outs.append(G.rollout(sb['P0'], sb['BL0'], sb['meta'])[0].float().cpu().numpy())
    return np.concatenate(outs)[:, None]                          # [N,1,22,25,2]


@torch.no_grad()
def predict_samples(G, D, T, S, bs=64, swap=False, seed=0):
    G.eval()
    D.eval()
    torch.manual_seed(seed)
    outs = []
    for sl in _slices(len(T['b']), bs):
        sb = _batch(T, sl, swap)
        bb = G.rollout(sb['P0'], sb['BL0'], sb['meta'])[0]
        outs.append(D.sample(sb['P0'], sb['BL0'], sb['meta'], bb, S=S).cpu().numpy().astype(np.float16))
    return np.concatenate(outs)                                    # [N,S,22,25,2]


def mean_ade(pred, d):
    err = np.linalg.norm(pred.astype(np.float32) - d['pos'][:, None, :, H + 1:], axis=-1).mean(-1)
    return float(err.mean(1)[d['mask']].mean())


def b_effect(p, p_sw, d):
    """b를 뒤집었을 때 1초 뒤 위치가 얼마나 달라지나(m), passer와의 거리 구간별."""
    disp = np.linalg.norm(p[:, 0, :, -1] - p_sw[:, 0, :, -1], axis=-1)
    p_t = d['pos'][:, :, H]
    dist = np.linalg.norm(p_t - p_t[:, :1], axis=-1)
    out = {'passer': float(disp[:, 0][d['mask'][:, 0]].mean())}
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, np.inf)]:
        m = d['mask'] & (dist > 0) & (dist >= lo) & (dist < hi)
        out[f'{lo}-{hi}m'] = float(disp[m].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--feat', choices=['bdist', 'rel', 'rel_terrain', 'rel_bd3', 'rel_lean'], default='bdist')
    ap.add_argument('--ar', action='store_true')
    ap.add_argument('--weighted', action='store_true')
    ap.add_argument('--lam_traj', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs_g', type=int, default=80)
    ap.add_argument('--epochs_d', type=int, default=60)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--S_test', type=int, default=20)
    ap.add_argument('--bb_only', action='store_true', help='잔차 디퓨전 없이 적분 궤적만 학습·평가')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: use only the first N scenes per split')
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--runs_dir', default='./data/dfl/intent_gen/runs')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda'
    out = os.path.join(args.runs_dir, args.run)
    os.makedirs(out, exist_ok=True)
    data = {s: harness.load_split(args.sample_dir, s) for s in ('train', 'val', 'test')}
    if args.limit:
        data = {s: {k: v[:args.limit] for k, v in d.items()} for s, d in data.items()}
    T = {s: to_tensors(d, dev) for s, d in data.items()}
    n = len(T['train']['b'])
    log = {'args': vars(args), 'stage1': [], 'stage2': []}
    t0 = time.time()

    # ---- 1단계: 적분 궤적 ---------------------------------------------------------------------------
    G = Backbone(feat=args.feat, ar=args.ar).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs_g * len(_slices(n, args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    best = float('inf')
    for ep in range(args.epochs_g):
        G.train()
        perm = torch.randperm(n, device=dev)
        agg = {}
        for sl in _slices(n, args.bs):
            bt = split_batch({k: v[perm[sl]] for k, v in T['train'].items()}, flip=bool(torch.rand(1) < 0.5))
            loss, parts = G.loss(bt, weighted=args.weighted, lam_traj=args.lam_traj)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v) * (sl.stop - sl.start) / n
        rec = {'epoch': ep + 1, **{k: round(v, 4) for k, v in agg.items()}, 'min': round((time.time() - t0) / 60, 1)}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs_g:
            rec['val_ADE'] = round(mean_ade(predict_bb(G, T['val']), data['val']), 4)
            if rec['val_ADE'] < best:
                best = rec['val_ADE']
                torch.save({'model': G.state_dict(), 'feat': args.feat, 'ar': args.ar, 'epoch': ep + 1},
                           os.path.join(out, 'g.pt'))
        log['stage1'].append(rec)
        print(json.dumps({'stage': 1, **rec}, ensure_ascii=False), flush=True)
    ck = torch.load(os.path.join(out, 'g.pt'), map_location=dev)
    G.load_state_dict(ck['model'])
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    log['g_best_epoch'], log['g_val_ADE'] = ck['epoch'], best
    if args.bb_only:
        d = data['test']
        bb, bb_sw = predict_bb(G, T['test']), predict_bb(G, T['test'], swap=True)
        np.save(os.path.join(out, 'pred_bb_test.npy'), bb.astype(np.float32))
        np.save(os.path.join(out, 'pred_bb_test_bswap.npy'), bb_sw.astype(np.float32))
        pd.concat([harness.evaluate(bb, d, args.run + '_bb'), harness.evaluate(bb_sw, d, args.run + '_bb_bswap')]
                  ).to_csv(os.path.join(out, 'test_metrics.csv'), index=False)
        log['b_effect_bb'] = b_effect(bb, bb_sw, d)
        log['minutes'] = round((time.time() - t0) / 60, 1)
        with open(os.path.join(out, 'log.json'), 'w') as fh:
            json.dump(log, fh, ensure_ascii=False, indent=1)
        print('done', args.run, json.dumps({'g_best_epoch': ck['epoch'], 'g_val_ADE': best, 'min': log['minutes']}),
              flush=True)
        return

    # ---- 2단계: 잔차 디퓨전(적분 궤적 고정) -----------------------------------------------------------
    with torch.no_grad():
        sq, cnt = torch.zeros(5, device=dev), 0
        for sl in _slices(n, 256):
            sb = _batch(T['train'], sl)
            bb = G.rollout(sb['P0'], sb['BL0'], sb['meta'])[0]
            res = (sb['fut'] - bb)[:, :, STEP - 1::STEP]                  # [B,22,5,2]
            m = sb['meta']['mask']
            sq += (res[m] ** 2).mean(-1).sum(0)
            cnt += int(m.sum())
        res_std = (sq / cnt).sqrt()
    log['res_std'] = [round(float(x), 4) for x in res_std]
    D = ResDiff(feat=args.feat).to(dev)
    D.res_std.copy_(res_std)
    opt = torch.optim.AdamW(D.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs_d * len(_slices(n, args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    for ep in range(args.epochs_d):
        D.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for sl in _slices(n, args.bs):
            bt = split_batch({k: v[perm[sl]] for k, v in T['train'].items()}, flip=bool(torch.rand(1) < 0.5))
            with torch.no_grad():
                bb = G.rollout(bt['P0'], bt['BL0'], bt['meta'])[0]
            loss = D.loss(bt, bb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * (sl.stop - sl.start) / n
        rec = {'epoch': ep + 1, 'diff': round(tot, 4), 'min': round((time.time() - t0) / 60, 1)}
        if (ep + 1) % 20 == 0 or ep + 1 == args.epochs_d:
            pv = predict_samples(G, D, T['val'], S=5)
            rec['val_ADE_sample'] = round(mean_ade(pv, data['val']), 4)
            rec['val_ADE_mean'] = round(mean_ade(pv.astype(np.float32).mean(1, keepdims=True), data['val']), 4)
        log['stage2'].append(rec)
        print(json.dumps({'stage': 2, **rec}, ensure_ascii=False), flush=True)
    torch.save({'model': D.state_dict(), 'feat': args.feat}, os.path.join(out, 'd.pt'))

    # ---- 시험 경기 --------------------------------------------------------------------------------
    d = data['test']
    bb, bb_sw = predict_bb(G, T['test']), predict_bb(G, T['test'], swap=True)
    ps, ps_sw = (predict_samples(G, D, T['test'], args.S_test),
                 predict_samples(G, D, T['test'], args.S_test, swap=True))
    np.save(os.path.join(out, 'pred_bb_test.npy'), bb.astype(np.float32))
    np.save(os.path.join(out, 'pred_bb_test_bswap.npy'), bb_sw.astype(np.float32))
    np.save(os.path.join(out, 'pred_test.npy'), ps)
    res = [harness.evaluate(bb, d, args.run + '_bb'), harness.evaluate(bb_sw, d, args.run + '_bb_bswap'),
           harness.evaluate(ps.astype(np.float32), d, args.run),
           harness.evaluate(ps_sw.astype(np.float32), d, args.run + '_bswap')]
    pd.concat(res).to_csv(os.path.join(out, 'test_metrics.csv'), index=False)
    log['physics'] = [harness.physics(bb, d, args.run + '_bb'), harness.physics(ps.astype(np.float32), d, args.run)]
    log['b_effect_bb'] = b_effect(bb, bb_sw, d)
    log['minutes'] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(out, 'log.json'), 'w') as fh:
        json.dump(log, fh, ensure_ascii=False, indent=1)
    print('done', args.run, json.dumps({'g_best_epoch': ck['epoch'], 'g_val_ADE': best,
                                        'b_effect_bb': log['b_effect_bb'], 'min': log['minutes']}), flush=True)


if __name__ == '__main__':
    main()
