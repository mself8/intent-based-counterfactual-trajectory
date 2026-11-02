# Intent-conditioned counterfactual trajectories (PAUSA follow-up)

PAUSA values pass timing by asking how a play would have unfolded if the pass had been made at a different moment.
To answer that, it needs the next second of player movement *without* the pass, and it generates that second by moving every player at constant velocity.
This folder replaces the constant-velocity assumption with learned trajectory generators and tests them against real movement.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Builds scene samples from the 7 DFL matches: 2 s of history and the next 1 s at 25 Hz. "Pass" scenes start one frame before a PAUSA pass; "no pass" scenes are moments where the ball carrier kept the ball for at least one more second. |
| `model.py` | IAD (Liu et al., AAAI 2026) adapted to soccer. All 22 players are generated jointly (player–player attention, ball as a context token), conditioned on short-term intent (acceleration direction and magnitude) and endpoint candidates, with DDIM sampling. `--cond b_dist` also feeds the pass state and the distance to the ball carrier. |
| `train.py` | Training and evaluation. The checkpoint with the lowest validation ADE is evaluated on a held-out test match. |
| `harness.py` | Evaluation harness that compares generated and real 1-second futures for both scene types. |
| `report.py` | Collects baseline and model metrics into tables. |
| `animate_scene.py` | Side-by-side animation of one scene: real movement, the constant-velocity baseline and generated trajectories. |

## Findings so far

- Ablations show that the diffusion model largely ignores its intent condition.
- Integrating the intent predictor's outputs alone gives trajectories closer to the real movement of defenders near the ball carrier than both the constant-velocity baseline and the diffusion model.

This is ongoing research.

## Usage

Run from the repository root after preparing the ELASTIC-synchronized DFL data as described in the main README.

```bash
python intent_gen/build_dataset.py --data_dir ./data/dfl/elastic --out_dir ./data/dfl/intent_gen/samples
python intent_gen/train.py --cond past --run past_s0
python intent_gen/train.py --cond b_dist --run bdist_s0
python intent_gen/report.py --runs past_s0 bdist_s0
python intent_gen/animate_scene.py --game DFL-MAT-J03WMX --frame 43680 --run bdist_s0
```

Samples, checkpoints and animations are written to `data/dfl/intent_gen/`, which is not tracked.
Minimal dependencies are listed in `requirements-min.txt`.
