"""IAD 축구판 학습·평가.

    CUDA_VISIBLE_DEVICES=0 nice -n 19 /opt/conda/bin/python intent_gen/train.py --cond past --run past_s0
    CUDA_VISIBLE_DEVICES=0 nice -n 19 /opt/conda/bin/python intent_gen/train.py --cond b_dist --run bdist_s0

검증 경기에서 표본 하나의 평균 오차(ADE)가 가장 낮은 체크포인트로 시험 경기를 평가한다.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from model import H, IADSoccer, prepare  # noqa: E402

KEYS = ['pos', 'ball', 'mask', 'team', 'gk', 'b']


def to_tensors(d, device):
    return {k: torch.as_tensor(d[k].astype(np.int64) if d[k].dtype == np.int8 else d[k], device=device)
            for k in KEYS}


def mean_ade(pred, d):
    err = np.linalg.norm(pred - d['pos'][:, None, :, H + 1:], axis=-1).mean(-1)   # [N,S,22]
    return float(err.mean(1)[d['mask']].mean())


@torch.no_grad()
def predict(model, t, dv_std, S, bs=256, swap_b=False):
    model.eval()
    outs = []
    for i in range(0, len(t['b']), bs):
        bt = {k: v[i:i + bs] for k, v in t.items()}
        if swap_b:
            bt['b'] = 1 - bt['b']
        pos, _ = model.sample(prepare(bt, dv_std), dv_std, S=S)
        outs.append(pos.cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=['past', 'b_dist'], required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--eval_every', type=int, default=10)
    ap.add_argument('--S_val', type=int, default=5)
    ap.add_argument('--S_test', type=int, default=20)
    ap.add_argument('--sample_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--runs_dir', default='./data/dfl/intent_gen/runs')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda'
    out = os.path.join(args.runs_dir, args.run)
    os.makedirs(out, exist_ok=True)
    data = {s: harness.load_split(args.sample_dir, s) for s in ('train', 'val', 'test')}
    T = {s: to_tensors(d, dev) for s, d in data.items()}

    with torch.no_grad():
        x0 = prepare(T['train'], 1.0)['x0']
        dv_std = float(x0[T['train']['mask']].std())
    model = IADSoccer(cond=args.cond).to(dev)
    n = len(T['train']['b'])
    steps = args.epochs * ((n + args.bs - 1) // args.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    log = {'args': vars(args), 'dv_std': dv_std, 'n_params': sum(p.numel() for p in model.parameters()),
           'epochs': []}
    best, t0 = float('inf'), time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        agg = {}
        for i in range(0, n, args.bs):
            bt = {k: v[perm[i:i + args.bs]] for k, v in T['train'].items()}
            loss, parts = model.loss(prepare(bt, dv_std, flip_y=bool(torch.rand(1) < 0.5)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v) * len(bt['b']) / n
        rec = {'epoch': ep + 1, **{k: round(v, 4) for k, v in agg.items()}, 'min': round((time.time() - t0) / 60, 1)}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            rec['val_ADE'] = round(mean_ade(predict(model, T['val'], dv_std, args.S_val), data['val']), 4)
            if rec['val_ADE'] < best:
                best = rec['val_ADE']
                torch.save({'model': model.state_dict(), 'dv_std': dv_std, 'cond': args.cond, 'epoch': ep + 1},
                           os.path.join(out, 'best.pt'))
        log['epochs'].append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    ck = torch.load(os.path.join(out, 'best.pt'), map_location=dev)
    model.load_state_dict(ck['model'])
    log['best_epoch'] = ck['epoch']
    pred = predict(model, T['test'], dv_std, args.S_test)
    np.save(os.path.join(out, 'pred_test.npy'), pred.astype(np.float32))
    res = [harness.evaluate(pred, data['test'], args.run)]
    log['physics'] = harness.physics(pred, data['test'], args.run)
    if args.cond == 'b_dist':   # same scenes, b flipped
        pred_sw = predict(model, T['test'], dv_std, args.S_test, swap_b=True)
        np.save(os.path.join(out, 'pred_test_bswap.npy'), pred_sw.astype(np.float32))
        res.append(harness.evaluate(pred_sw, data['test'], args.run + '_bswap'))
    import pandas as pd
    pd.concat(res).to_csv(os.path.join(out, 'test_metrics.csv'), index=False)
    with open(os.path.join(out, 'log.json'), 'w') as fh:
        json.dump(log, fh, ensure_ascii=False, indent=1)
    print('done', args.run, 'best_epoch', ck['epoch'], 'val_ADE', best, flush=True)


if __name__ == '__main__':
    main()
