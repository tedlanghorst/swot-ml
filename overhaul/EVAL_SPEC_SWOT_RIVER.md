# Experiment W: amplitude-free SWOT terms in stage-2 training (written 2026-11-12, before any arm was scored)

Addendum to `overhaul/EVAL_SPEC.md`. It lives in the worktree branch `worktree-swot-river-rank` because this session
could not edit the shared `overhaul/` directory; the commit time is the record that the rule came first.

## Why
- User, 2026-11-12: revisit SWOT as a training target; use river channels as a directional signal; train a
  multi-objective model whose routed output also predicts what SWOT sees. "Set a noise floor of 0.5 (we assume the
  ceiling is no lower than 0.5); we don't want it to learn to just ignore that." Go for #1, #2, then #3 on my
  judgement, unattended.
- The earlier SWOT storage target (c2wvas) failed: a squared error on storage anomalies rewarded damping the pools
  (sd of storage change 0.27 of observed vs 0.74) and cost 0.051 NSE below reservoirs.
- Diagnostics of the same day (`stage2/diag/swot_rank_f0/`): the rank agreement of routed flow with SWOT elevation
  follows gauge rank skill in sign about 80% of the time; ceiling (observed gauge flow vs elevation) median 0.75,
  Colorado 0.51. Offline head fit (`swot_head_cv.txt`): elevation = a + b sqrt(flow) predicts held-out passes as well
  as a free monotone curve (median r 0.655 vs 0.647; log 0.625), so the head needs no free shape.

## Terms (train_s2.py of the worktree; defaults = old runs)
- River term, `--rsw=1`: on SWOT-era windows, per plain river SWORD reach (type 1, lakeflag 0, >= 20 observed days,
  >= 10 in the window), rho = Pearson correlation of sqrt(routed flow) and SWOT WSE. Loss per reach
  (1 - rho) / s + log s, with a learned per-reach noise scale s in [0.1, 0.5]. s <= 0.5 is the user's floor: a
  reach's ceiling is taken to be at least 0.5, so no reach can be down-weighted by more than 5x against the
  cleanest ones. Reaches with MERIT comid % 5 == 0 are held out of the term (scored only).
- Reservoir term, `--swot=1 --swot_form=corr`: per dam, 1 - correlation of modelled and SWOT LakeSP storage on
  the observed days of the window. It cannot be lowered by damping.
- As in c2wvas: a SWOT-era window (probability 0.25) carries the SWOT terms and the penalties only; no gauge after
  2022-12-31 enters any loss. Weight 1 for both terms, not tuned.

## Arms (c2wva settings otherwise; fold 0; seeds 0 and 1; control c2wva, c2wva_s1)
- WR = river term; WS = reservoir term; WRS = both.

## Scoring
- `ladder_diag --spin=cont --test=1` + `itp_report.py` on 1980-2022 (fold-0 validation + test gauges, ResOpsUS) ->
  `stage2/diag/w_f0/report.txt`.
- `swot_era_flows.py --local=1` + `swot_rank.py`, `swot_local.py`, `swot_era_gauges.py` on the SWOT period ->
  `stage2/diag/w_swot_f0/`.

## Rule (each arm against the control, means over the two seeds)
Primary, at least one:
1. Held-out SWOT reaches (comid % 5 == 0; plain river): mean paired change of the rank score rho(model flow, WSE),
   arm - control, >= +0.02 with the node-bootstrap 95% CI above 0. (For WS no river reach is in a loss; the same
   held-out set is used.)
2. Loss timing on 1980-2022, as the rule of experiments I / IF / T: spurious loss at near-natural gauges
   <= 2/3 x control, or within-gauge rho(monthly loss share, natural flow) at depleted < 0.9 gauges
   <= control - 0.2.
3. Reservoirs (WS, WRS), as experiment R: annual-range ratio at dams >= 1000 MCM >= control + 0.10 and Missouri
   main stem >= 2 x control.

Guards, all required (itp_report's): median NSE of all gauges >= control - 0.01; depleted < 0.9 >= control - 0.02;
regulated main stems' median |log beta| <= control + 0.05; plus below-reservoir median NSE >= control - 0.02 and
ResOpsUS median r_dS >= control - 0.02 (the c2wvas failure).

A primary + all guards -> PROMISING: candidate for folds 1-4 (seed 0), which waits for the user. Otherwise a negative
result. One method for all basins: the rule is read on the pooled numbers; basin lines are context.

Reported, not scored: gauge NSE in 2023-07+ (the EVAL_SPEC test window; never used to choose), the local toggle,
CU&L, the learned noise scales, in-sample SWOT fits.

## Known limits, stated before the runs
- Neither term can inform the mean loss volume: a constant flow bias leaves the correlation unchanged.
- Gauges (<= 2022) and SWOT (2023-07+) never share a window; the terms meet only through the shared weights.
- The floor of 0.5 is above the measured ceiling at about half of the gauged Colorado reaches (observed-flow
  held-out r below 0.5 at 52%), so there the term pushes against noise by design of the user's choice; the held-out
  reaches and the guards are the check that this does no harm.

## Outcome (2026-11-13; stage2/diag/w_f0/report.txt, stage2/diag/w_swot_f0/)
Rule read on arm means over two seeds against the control. No arm is PROMISING.

| | WR | WS | WRS | control |
|---|---|---|---|---|
| Primary 1: held-out SWOT reaches, rho arm - control | +0.053 [+0.043, +0.065] pass | -0.004 fail | +0.056 [+0.045, +0.067] pass | |
| Primary 2: rho(loss share, flow), west mean | 0.279 pass | 0.399 fail | 0.244 pass | 0.480 |
| Primary 2: spurious loss, west mean | 0.056 fail | 0.058 fail | 0.061 fail | 0.070 |
| Primary 3: annual-range ratio, dams >= 1000 MCM | n/a | 0.721 fail | 0.575 fail | 0.740 |
| Guard: median NSE, all gauges | 0.616 ok | 0.619 ok | 0.616 ok | 0.615 |
| Guard: NSE depleted < 0.9 (>= 0.399) | 0.396 FAIL | 0.384 FAIL | 0.406 ok | 0.419 |
| Guard: regulated main stems' abs log beta (<= 0.235) | 0.300 FAIL | 0.134 ok | 0.270 FAIL | 0.185 |
| Guard: NSE below reservoirs (>= 0.326) | 0.364 ok | 0.317 FAIL | 0.373 ok | 0.346 |
| Guard: ResOpsUS r_dS (>= 0.617) | 0.618 ok | 0.652 ok | 0.669 ok | 0.637 |

WRS passes two primaries and misses one guard (volume bias on regulated main stems, 1980-2022). Seed spread of that
guard: control 0.130 / 0.239, WRS 0.278 / 0.261, WR 0.397 / 0.203.
Reported only: gauges 2023-07+ (test window), WRS - control: all +0.003, below reservoirs +0.044 [+0.023, +0.070],
regulated main stems +0.095 [+0.064, +0.168] (abs log beta 0.301 vs 0.503); WS: regulated main stems -0.127. JAS / AMJ loss
ratio 0.15-0.25 with the river term vs 0.05 (apparent ~0.3). Learned noise scale, median: 0.30 Mississippi, 0.31
Colorado, 0.33 Columbia, 0.20 ACF, 0.18 Susquehanna. Folds 1-4 not run.
