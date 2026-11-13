"""Local toggle (2026-11-12, user go "#2"): inside a gauged reach, with the OBSERVED flow of the nearest upstream gauges
fed in, do the model's losses on that reach alone bring the flow closer (in rank) to the SWOT elevation?
Fixes the confound of swot_rank.py section 5, where "losses off" switched off every loss upstream as well.

Inputs: swot_era_flows.py --local=1 output (--diag). At a SWOT node n with upstream gauges (no model dam between):
  local_on  = V_full(n)   - routed model flow of the upstream gauges (with the path pass-through) + the same from observations
  local_off = V_noloss(n) - routed model flow of the upstream gauges (no losses)                 + the same from observations
so both carry the observed inflow and the model's local runoff; they differ only by the losses between the gauges
and n (on the through-flow and on the local runoff). Linear correction: capped withdrawals keep the pass-through of
the free run (as REACH_TABLES_FOR_SWOT.md). Days with an unobserved upstream day in a kernel are dropped.
Scores per plain river reach (>= 10 usable days): Spearman rho with WSE of local_on, local_off, the free run and the
routed observed inflow alone; loc_loss = 1 - mean(local_on) / mean(local_off); up_share = routed observed inflow /
mean(local_off). 2023+ gauge flows enter as inputs of a diagnostic only (nothing is trained or chosen on it).
Statistics over nodes; CIs resample gauged reaches (clusters = the first reach-table gauge at or below the node).
Writes --diag/swot_local.parquet, swot_local.txt.
Usage: python swot_local.py --diag=DIR --control=c2wva,c2wva_s1 --arms=I:I,I_s1;IF:IF,IF_s1;T:T,T_s1
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
DIAG = Path(opt["diag"])
OV = Path("/project/swot-ml/data/overhaul")
CONTROL = opt.get("control", "c2wva,c2wva_s1").split(",")
ARMS = {a.split(":")[0]: a.split(":")[1].split(",") for a in opt.get("arms", "").split(";") if a}
RT = Path(opt.get("reach_table", "/project/swot-ml/swot-ml/overhaul/stage2/diag/reachinc_c2wva_f0/gauges.parquet"))
MIN_DAYS = 10
BNAME = {"USGS-07289000": "Mississippi", "USGS-09521100": "Colorado", "USGS-14246900": "Columbia",
         "USGS-02359170": "ACF", "USGS-01578310": "Susquehanna"}
rng = np.random.default_rng(0)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def spearman(x, y):
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) >= 3 and rx.std() > 0 and ry.std() > 0 else np.nan


def cboot(v, cl, n=1000):
    """Mean of v with a 95% CI from resampling clusters cl."""
    df = pd.DataFrame({"v": np.asarray(v, float), "c": np.asarray(cl)}).dropna()
    if df.c.nunique() < 4:
        return (df.v.mean() if len(df) else np.nan), np.nan, np.nan, len(df), df.c.nunique()
    g = df.groupby("c").v.agg(["sum", "size"])
    ii = rng.integers(0, len(g), (n, len(g)))
    m = g["sum"].values[ii].sum(1) / g["size"].values[ii].sum(1)
    return df.v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(df), len(g)


RM = pd.read_parquet(OV / "swot_rivers/reach_match.parquet")
RM = RM[(RM.why == "ok") & (RM.type == 1) & (RM.lakeflag == 0)].set_index("reach_id")
OBS = pd.read_parquet(OV / "swot_rivers/reach_obs.parquet", columns=["reach_id", "date", "wse"])
OBS = OBS.groupby(["reach_id", "date"], as_index=False).wse.mean()
rows, runs = [], None
for basin, bn in BNAME.items():
    z = np.load(DIAG / f"flows_{basin}.npz", allow_pickle=True)
    if "tf_n_up" not in z.files:
        say(f"{bn}: no local pairs")
        continue
    dates = pd.DatetimeIndex(z["dates"])
    dpos = pd.Series(np.arange(len(dates)), index=dates)
    nodes, isg = z["nodes"].astype(str), z["is_gauge"]
    npos = {n: i for i, n in enumerate(nodes) if not isg[i]}
    n_up, bad = z["tf_n_up"], z["tf_bad"]
    runs = sorted({k.split("__")[0] for k in z.files if k.endswith("__full__rm")})
    A = {k: z[k] for k in z.files if "__" in k}
    rm = RM[RM.basin == basin]
    ob = OBS[OBS.reach_id.isin(rm.index) & OBS.date.isin(dates)]
    for rid, o in ob.groupby("reach_id"):
        r = rm.loc[rid]
        j = npos.get(r.node)
        if j is None or n_up[j] == 0:
            continue
        ti = dpos.reindex(o.date).values.astype(int)
        ok = ~bad[ti, j]
        if ok.sum() < MIN_DAYS:
            continue
        ti, w = ti[ok], o.wse.values[ok]
        row = {"basin": bn, "reach_id": rid, "node": r.node, "uparea": r.merit_uparea, "n": int(ok.sum()), "n_up": int(n_up[j])}
        for rn in runs:
            on = A[f"{rn}__full"][ti, j] - A[f"{rn}__full__rm"][ti, j] + A[f"{rn}__full__ro"][ti, j]
            off = A[f"{rn}__noloss"][ti, j] - A[f"{rn}__noloss__rm"][ti, j] + A[f"{rn}__noloss__ro"][ti, j]
            inflow = A[f"{rn}__noloss__ro"][ti, j]
            row.update({f"on__{rn}": spearman(on, w), f"off__{rn}": spearman(off, w), f"free__{rn}": spearman(A[f"{rn}__full"][ti, j], w),
                        f"in__{rn}": spearman(inflow, w), f"loc_loss__{rn}": 1 - on.mean() / max(off.mean(), 1e-9),
                        f"up_share__{rn}": inflow.mean() / max(off.mean(), 1e-9)})
        rows.append(row)
R = pd.DataFrame(rows)
# gauged reach of each node (cluster) and the observed / modelled change of that reach, 1980-2022 (reach table)
T = pd.read_parquet(RT)
T = T[(T.run == CONTROL[0]) & T.inc_obs.notna() & (T.tf_omean > 0)].drop_duplicates("gauge").set_index("gauge")
T["obs_ch"] = T.inc_obs / T.tf_omean
down = {e["source"]: e["target"] for e in json.load(open(OV / "ds_targets/graph.json"))["edges"]}
cache = {}


def reach_gauge(n):
    path = []
    while n is not None and n not in cache and n not in T.index:
        path.append(n)
        n = down.get(n)
    g = None if n is None else (n if n in T.index else cache[n])
    for p in path:
        cache[p] = g
    return g


R["rgauge"] = [reach_gauge(n) for n in R.node]
R["obs_ch"] = R.rgauge.map(T.obs_ch)
R["cluster"] = R.rgauge.fillna(R.node)
R.to_parquet(DIAG / "swot_local.parquet", index=False)
N = R.groupby(["basin", "node"]).agg({**{c: "mean" for c in R.columns if "__" in c}, "obs_ch": "first", "cluster": "first", "uparea": "first"}).reset_index()
c0 = CONTROL[0]
say(f"local toggle: {len(R)} plain river reaches on {len(N)} nodes with an upstream gauge feeding observed flow "
    f"(by basin {N.basin.value_counts().to_dict()}); usable days median {R.n.median():.0f}; "
    f"observed inflow share of the node's flow, median {N[f'up_share__{c0}'].median():.2f}")


def mean_of(names, pre):
    return N[[f"{pre}__{n}" for n in names if f"{pre}__{n}" in N.columns]].mean(1)


def line(lab, d, m):
    out = []
    for bn in ["Colorado", "Columbia", "Mississippi", "all"]:
        mm = m if bn == "all" else m & (N.basin == bn)
        a, lo, hi, n, k = cboot(d[mm], N.cluster[mm])
        if n >= 5:
            out.append(f"{bn} {a:+.3f} [{lo:+.3f}, {hi:+.3f}] (nodes {n}, reaches {k}, {np.mean(d[mm].dropna() > 0):.0%} up)")
    say(f"  {lab}: " + "; ".join(out))


say("\n## 1. Rank score with WSE, median over nodes (free run | observed inflow + local, losses on | losses off | observed inflow alone)")
for nm_, names in [("control", CONTROL)] + list(ARMS.items()):
    if all(f"on__{n}" in N.columns for n in names):
        say(f"  {nm_:8s} {mean_of(names, 'free').median():.3f} | {mean_of(names, 'on').median():.3f} | {mean_of(names, 'off').median():.3f} | {mean_of(names, 'in').median():.3f}")
say("\n## 2. Local losses on - off (mean over nodes, CI over gauged reaches), by the run's own local loss at the node")
for nm_, names in [("control", CONTROL)] + list(ARMS.items()):
    if not all(f"on__{n}" in N.columns for n in names):
        continue
    d, ll = mean_of(names, "on") - mean_of(names, "off"), mean_of(names, "loc_loss")
    say(f"{nm_}: nodes with local loss >= 10%: {int((ll >= 0.10).sum())}, 2-10%: {int(((ll >= 0.02) & (ll < 0.10)).sum())}, < 2%: {int((ll < 0.02).sum())}")
    line("local loss >= 10%", d, ll >= 0.10)
    line("local loss 2-10% ", d, (ll >= 0.02) & (ll < 0.10))
    line("local loss >= 2%, observed reach loses >= 2% (1980-2022)", d, (ll >= 0.02) & (N.obs_ch <= -0.02))
    line("local loss >= 2%, observed reach does not lose      ", d, (ll >= 0.02) & (N.obs_ch > 0))
say("\n## 3. Arms against the control, losses on, same observed inflow (does the arm's local loss fit SWOT better?)")
for nm_, names in ARMS.items():
    if all(f"on__{n}" in N.columns for n in names):
        d = mean_of(names, "on") - mean_of(CONTROL, "on")
        ll = np.maximum(mean_of(names, "loc_loss"), mean_of(CONTROL, "loc_loss"))
        line(f"{nm_:3s} all nodes            ", d, ll > -9)
        line(f"{nm_:3s} either local loss >= 2%", d, ll >= 0.02)
        line(f"{nm_:3s} observed reach loses  ", d, N.obs_ch <= -0.02)
say("\n## 4. What the observed inflow adds (losses on - free run), control")
line("all nodes", mean_of(CONTROL, "on") - mean_of(CONTROL, "free"), N.uparea > 0)
say("\n## 5. Where the observations lose water but the control's local loss is < 2%: does MORE loss fit SWOT? (not testable by a toggle;"
    " level only) median rank score on | off: "
    + str({bn: (round(mean_of(CONTROL, 'on')[m].median(), 3), int(m.sum())) for bn in ["Colorado", "Columbia", "Mississippi"]
           for m in [(N.basin == bn) & (N.obs_ch <= -0.02) & (mean_of(CONTROL, 'loc_loss') < 0.02)]}))
(DIAG / "swot_local.txt").write_text("\n".join(lines) + "\n")
