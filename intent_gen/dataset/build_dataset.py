"""DFL 7경기 -> 의도 조건 궤적 생성용 장면 표본.

장면 = 기준 프레임 t의 과거 2초(t 포함 51프레임) + 미래 1초(t+1..t+25), 25Hz.
  패스 O (b=1): PAUSA 패스 집합의 pass_frame - 1. PAUSA 가상 궤적이 출발하는 프레임과 같다.
  패스 X (b=0): 공 소유 구간(receive / interception / ball_recovery -> 다음 이벤트) 안에서
               t+25가 구간 끝보다 앞서는 프레임을 5프레임 간격으로.
좌표·속도는 PAUSA 로더 그대로(중심 좌표, m)이고, 공을 가진 팀이 +x로 공격하도록 180도 돌린다.

    python intent_gen/dataset/build_dataset.py --data_dir ./data/dfl/elastic --out_dir ./data/dfl/intent_gen/samples
    FUT_FRAMES=50 python intent_gen/dataset/build_dataset.py --out_dir ./data/dfl/intent_gen/samples_2s   # 미래 2초
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'pausa'))
import loader as io  # noqa: E402

GAMES = ['DFL-MAT-J03WMX', 'DFL-MAT-J03WN1', 'DFL-MAT-J03WOH', 'DFL-MAT-J03WOY',
         'DFL-MAT-J03WPY', 'DFL-MAT-J03WQQ', 'DFL-MAT-J03WR9']
H, F = 50, int(os.environ.get('FUT_FRAMES', 25))   # past frames before t, future frames after t
STRIDE_HOLD = 5
PASS_EVENTS = ['pass', 'cross']
GAIN_EVENTS = ['receive', 'interception', 'ball_recovery']
MAX_PER_TEAM = 11
HOLD_DIST_T, HOLD_DIST_MAX = 2.0, 3.0  # m, ball-carrier distance at t / max over the future window
MAX_SPEED = 12.0                       # m/s, same outlier bound as loader.calc_player_velocities


def load_game(data_dir, game_id):
    """calculate_obso.run_obso_virtual 와 같은 순서로 트래킹·이벤트·패스 집합을 만든다."""
    ld = io.ElasticLoader(data_dir, game_id)
    events = ld.get_event_data()
    home = ld.calc_player_velocities(ld.get_trace_data('Home'))
    away = ld.calc_player_velocities(ld.get_trace_data('Away'))
    ev = ld.convert_common_format_for_event(events, home, away)
    ev = ld.check_home_away_event(ev, home, away)
    passes = ev[
        ev['type_name'].isin(PASS_EVENTS) & (ev['result_name'] == True) &  # noqa: E712
        (ev['team'] == ev['receiver_team']) & ev['receive_frame'].notna()
    ].reset_index(drop=True)
    direction = {'Home': ld.find_playing_direction(home, 'Home'),
                 'Away': ld.find_playing_direction(away, 'Away')}
    return home, away, ev, passes, direction


def stack_tracking(home, away):
    home = home.set_index('frame_id')
    away = away.set_index('frame_id')
    assert home.index.equals(away.index)
    codes = sorted(c[:-2] for c in home.columns if c.startswith('Home_') and c.endswith('_x')) + \
        sorted(c[:-2] for c in away.columns if c.startswith('Away_') and c.endswith('_x'))
    src = {c: (home if c.startswith('Home_') else away) for c in codes}
    pos = np.stack([np.stack([src[c][f'{c}_x'].values, src[c][f'{c}_y'].values], -1) for c in codes], 1)
    vel = np.stack([np.stack([src[c][f'{c}_vx'].values, src[c][f'{c}_vy'].values], -1) for c in codes], 1)
    ball = np.stack([home['ball_x'].values, home['ball_y'].values], -1)
    return (home.index.values.astype(np.int64), home['period_id'].values.astype(np.int64),
            codes, pos.astype(np.float32), vel.astype(np.float32), ball.astype(np.float32))


def hold_intervals(ev):
    """(period, carrier, start, end): 공을 얻은 프레임부터 다음 이벤트 프레임 직전까지."""
    out = []
    ev = ev.sort_values(['period_id', 'start_frame'], kind='stable')
    for period, g in ev.groupby('period_id'):
        rows = g[['type_name', 'player_code', 'start_frame']].to_records(index=False)
        for r, nxt in zip(rows[:-1], rows[1:]):
            if r.type_name in GAIN_EVENTS and isinstance(r.player_code, str):
                s, e = int(r.start_frame), int(nxt.start_frame)
                if e - s > F:
                    out.append((int(period), r.player_code, s, e))
    return out


def build_game(data_dir, game_id, gk_codes=None):
    home, away, ev, passes, direction = load_game(data_dir, game_id)
    frames, periods, codes, pos, vel, ball = stack_tracking(home, away)
    row_of = {f: i for i, f in enumerate(frames)}
    code_idx = {c: i for i, c in enumerate(codes)}
    # find_goalkeeper returns the shirt number only
    gk = {f'Home_{io.ElasticLoader.find_goalkeeper(home)}', f'Away_{io.ElasticLoader.find_goalkeeper(away)}'}

    anchors = []  # (t_frame, carrier_code, b, pass index in PAUSA pass set)
    for k, p in enumerate(passes.itertuples()):
        anchors.append((int(p.start_frame) - 1, p.player_code, 1, k))
    n_pass_total = len(anchors)
    for period, carrier, s, e in hold_intervals(ev):
        for t in range(s, e - F, STRIDE_HOLD):
            anchors.append((t, carrier, 0, -1))

    keep = {k: [] for k in ['pos', 'vel_pausa', 'ball', 'mask', 'team', 'gk', 'carrier', 'b', 'frame',
                            'period', 'pass_event_id', 'codes', 'dist_bc', 'dist_bc_max']}
    drop = {'window': 0, 'carrier_missing': 0, 'carrier_jump': 0, 'not_holding': 0}
    n_jump = 0
    for t, carrier, b, pid in anchors:
        i = row_of.get(t)
        if i is None or i - H < 0 or i + F >= len(frames) or frames[i + F] - frames[i - H] != H + F \
                or periods[i - H] != periods[i + F]:
            drop['window'] += 1
            continue
        sl = slice(i - H, i + F + 1)
        complete = ~np.isnan(pos[sl]).any(axis=(0, 2))           # [n_obj]
        atk = carrier.split('_')[0]
        if carrier not in code_idx or not complete[code_idx[carrier]]:
            drop['carrier_missing'] += 1
            continue
        team_slots = []
        for side in (atk, 'Away' if atk == 'Home' else 'Home'):
            on = [c for c in codes if c.startswith(side + '_') and complete[code_idx[c]]]
            if side == atk:  # carrier first in attacking block
                on = [carrier] + [c for c in on if c != carrier]
            team_slots.append((on + [None] * MAX_PER_TEAM)[:MAX_PER_TEAM])
        slots = team_slots[0] + team_slots[1]
        sgn = np.float32(direction[atk])  # +1: attacking team already attacks +x
        P = np.full((2 * MAX_PER_TEAM, H + F + 1, 2), np.nan, np.float32)
        V = np.full((2 * MAX_PER_TEAM, 2), np.nan, np.float32)
        for k, c in enumerate(slots):
            if c is not None:
                P[k] = pos[sl, code_idx[c]] * sgn
                V[k] = vel[i, code_idx[c]] * sgn
        # tracking teleports (> MAX_SPEED between frames): drop that player from the scene, or the scene if carrier
        jump = np.nanmax(np.linalg.norm(np.diff(P, axis=1), axis=-1), axis=1) / 0.04 > MAX_SPEED
        if jump[0]:
            drop['carrier_jump'] += 1
            continue
        P[jump] = np.nan
        V[jump] = np.nan
        slots = [None if j else c for c, j in zip(slots, jump)]
        n_jump += int(jump.sum())
        B = ball[sl] * sgn
        d_bc = np.linalg.norm(B - P[0], axis=-1)
        # 패스 X label = carrier really keeps the ball over the future window (events alone include deflections)
        if b == 0 and not (d_bc[H] <= HOLD_DIST_T and np.nanmax(d_bc[H:]) <= HOLD_DIST_MAX):
            drop['not_holding'] += 1
            continue
        keep['pos'].append(P)
        keep['vel_pausa'].append(V)
        keep['ball'].append(B)
        keep['mask'].append(np.array([c is not None for c in slots]))
        keep['team'].append(np.array([1] * MAX_PER_TEAM + [0] * MAX_PER_TEAM, np.int8))
        keep['gk'].append(np.array([c in gk for c in slots]))
        keep['carrier'].append(0)
        keep['b'].append(b)
        keep['frame'].append(t)
        keep['period'].append(int(periods[i]))
        keep['pass_event_id'].append(pid)
        keep['codes'].append(np.array([c or '' for c in slots], dtype='<U12'))
        keep['dist_bc'].append(float(d_bc[H]))
        keep['dist_bc_max'].append(float(np.nanmax(d_bc[H:])))

    arrs = {k: np.stack(v) if k in ('pos', 'vel_pausa', 'ball', 'mask', 'team', 'gk', 'codes')
            else np.array(v) for k, v in keep.items()}
    stats = {'game': game_id, 'passes_pausa': n_pass_total, 'n_pass_O': int((arrs['b'] == 1).sum()),
             'n_pass_X': int((arrs['b'] == 0).sum()), **{f'drop_{k}': v for k, v in drop.items()},
             'masked_jump_players': n_jump,
             'direction': direction}
    return arrs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/dfl/elastic')
    ap.add_argument('--out_dir', default='./data/dfl/intent_gen/samples')
    ap.add_argument('--games', nargs='*', default=GAMES)
    ap.add_argument('--n_jobs', type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def run(g):
        arrs, stats = build_game(args.data_dir, g)
        np.savez_compressed(os.path.join(args.out_dir, f'{g}.npz'), **arrs)
        return stats

    for s in Parallel(n_jobs=args.n_jobs)(delayed(run)(g) for g in args.games):
        print(s)


if __name__ == '__main__':
    main()
