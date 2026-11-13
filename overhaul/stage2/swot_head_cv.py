"""Offline head fit (2026-11-12, user go "#1"): how much of a reach's SWOT elevation can a monotone function of the
routed flow explain on passes it was not fitted on? No model is trained or changed: the saved SWOT-era flows of
swot_era_flows.py (--diag) are the inputs, one head per reach.

Per plain river reach (SWORD type 1, lakeflag 0) with >= MIN_DAYS observed days, the observed days are split
alternately into two halves; each head is fitted on one half and scored on the other (both ways; r = Pearson
correlation of predicted and observed WSE, mean of the two halves):
  log   h = a + b log(Q + q0), q0 = 1% of the reach's mean flow     (2 parameters)
  p0.3 / p0.5 / p1   h = a + b Q^p                                  (2 parameters)
  best  the p of {log, 0.2, 0.3, 0.4, 0.5, 0.7, 1} with the best fit on the training half   (3 parameters)
  iso   isotonic (any monotone curve), linear interpolation between the training points
b >= 0 is not enforced in the fit; a negative held-out r is a failure of the reach, kept as it is.
The choice of the head form for training is made on the MODEL flows only (--run, default the control). The same
fit with the OBSERVED flow of a gauge in the reach's unit catchment is reported as a check (2023+ gauge data, never
used for training or selection; nothing is chosen on it).
Writes --diag/swot_head_cv.parquet and swot_head_cv.txt.
Usage: python swot_head_cv.py --diag=DIR [--run=c2wva] [--others=b,I]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
DIAG = Path(opt["diag"])
OV = Path("/project/swot-ml/data/overhaul")
RUN = opt.get("run", "c2wva")
OTHERS = [r for r in opt.get("others", "b,I").split(",") if r]
MIN_DAYS = 20
PS = ["log", 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
BNAME = {"USGS-07289000": "Mississippi", "USGS-09521100": "Colorado", "USGS-14246900": "Columbia",
         "USGS-02359170": "ACF", "USGS-01578310": "Susquehanna"}
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def tr(q, p, q0):
    return np.log(q + q0) if p == "log" else np.power(np.maximum(q, 0.0), p)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def lin_fit_pred(x_tr, y_tr, x_te):
    if x_tr.std() == 0:
        return np.full(len(x_te), y_tr.mean())
    b, a = np.polyfit(x_tr, y_tr, 1)
    return a + b * x_te


def heads(q, w):
    """Held-out r of each head, mean over the two alternate halves."""
    q0 = 0.01 * max(q.mean(), 1e-9)
    idx = np.arange(len(q))
    out = {k: [] for k in ["log", "p0.3", "p0.5", "p1", "best", "iso"]}
    for A, Bm in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
        for nm, p in (("log", "log"), ("p0.3", 0.3), ("p0.5", 0.5), ("p1", 1.0)):
            out[nm].append(corr(lin_fit_pred(tr(q[A], p, q0), w[A], tr(q[Bm], p, q0)), w[Bm]))
        fit = [np.nan_to_num(corr(tr(q[A], p, q0), w[A]), nan=-1) for p in PS]
        pb = PS[int(np.argmax(fit))]
        out["best"].append(corr(lin_fit_pred(tr(q[A], pb, q0), w[A], tr(q[Bm], pb, q0)), w[Bm]))
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(q[A], w[A])
        out["iso"].append(corr(iso.predict(q[Bm]), w[Bm]))
    return {k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k, v in out.items()}


RM = pd.read_parquet(OV / "swot_rivers/reach_match.parquet")
RM = RM[(RM.why == "ok") & (RM.type == 1) & (RM.lakeflag == 0)].set_index("reach_id")
OBS = pd.read_parquet(OV / "swot_rivers/reach_obs.parquet", columns=["reach_id", "date", "wse", "wse_u"])
OBS = OBS.groupby(["reach_id", "date"], as_index=False).agg(wse=("wse", "mean"), wse_u=("wse_u", "mean"))
GG = pd.read_parquet(OV / "graph/gauges.parquet").drop_duplicates("gauge")
rows = []
for basin, bn in BNAME.items():
    z = np.load(DIAG / f"flows_{basin}.npz", allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    dpos = pd.Series(np.arange(len(dates)), index=dates)
    nodes, isg = z["nodes"].astype(str), z["is_gauge"]
    npos = {n: i for i, n in enumerate(nodes) if not isg[i]}
    gcol = {n: j for j, n in enumerate(nodes[isg])}
    fl = {r: z[f"{r}__full"] for r in [RUN] + OTHERS if f"{r}__full" in z.files}
    qobs = z["obs"]
    gg = GG[GG.gauge.isin(gcol) | GG.node.isin(gcol)]
    g_by = {}
    for g, c, a, nd in zip(gg.gauge, gg.comid, gg.area_merit, gg.node):
        g_by.setdefault(int(c), []).append((g if g in gcol else nd, a))
    rm = RM[RM.basin == basin]
    ob = OBS[OBS.reach_id.isin(rm.index) & OBS.date.isin(dates)]
    for rid, o in ob.groupby("reach_id"):
        r = rm.loc[rid]
        if r.node not in npos or len(o) < MIN_DAYS:
            continue
        ti, w = dpos.reindex(o.date).values.astype(int), o.wse.values
        row = {"basin": bn, "reach_id": rid, "node": r.node, "uparea": r.merit_uparea, "n": len(o),
               "wse_range": np.percentile(w, 95) - np.percentile(w, 5), "wse_u": float(o.wse_u.median()), "width": r.width}
        for rn, f in fl.items():
            for k, v in heads(f[ti, npos[r.node]].astype(float), w).items():
                row[f"{rn}__{k}"] = v
        for gname, ga in g_by.get(int(r.comid), []):
            if 0.8 <= ga / r.merit_uparea <= 1.02:
                qo = qobs[ti, gcol[gname]].astype(float)
                ok = np.isfinite(qo)
                if ok.sum() >= MIN_DAYS:
                    for k, v in heads(qo[ok], w[ok]).items():
                        row[f"obs__{k}"] = v
                    break
        rows.append(row)
R = pd.DataFrame(rows)
R.to_parquet(DIAG / "swot_head_cv.parquet", index=False)
H = ["log", "p0.3", "p0.5", "p1", "best", "iso"]
say(f"offline head fit: {len(R)} plain river reaches with >= {MIN_DAYS} observed days; median observed days {R.n.median():.0f}")
say(f"\n## held-out r of the WSE predicted from {RUN} flow, by head (median over reaches | share of reaches >= 0.5 | >= 0.8)")
for k in H:
    v = R[f"{RUN}__{k}"]
    say(f"  {k:5s} {v.median():.3f} | {(v >= 0.5).mean():.2f} | {(v >= 0.8).mean():.2f}")
say("\nmedian by basin:")
say(R.groupby("basin")[[f"{RUN}__{k}" for k in H]].median().round(3).rename(columns=lambda c: c.split("__")[1]).to_string())
R["size"] = pd.cut(R.uparea, [0, 1e4, 1e5, 1e7], labels=["<10k", "10k-100k", ">=100k"])
say("\nmedian by drainage area (km2):")
say(R.groupby("size", observed=False)[[f"{RUN}__{k}" for k in H]].median().round(3).rename(columns=lambda c: c.split("__")[1]).to_string())
for rn in OTHERS:
    if f"{rn}__log" in R.columns:
        say(f"\nrun {rn}: median held-out r " + str({k: round(R[f'{rn}__{k}'].median(), 3) for k in H}))
if "obs__log" in R.columns:
    G = R[R["obs__log"].notna()]
    say(f"\n## check with OBSERVED gauge flow ({len(G)} reaches with a gauge; reported only)")
    say("median held-out r: " + str({k: round(G[f'obs__{k}'].median(), 3) for k in H})
        + f"; same reaches with {RUN} flow: " + str({k: round(G[f'{RUN}__{k}'].median(), 3) for k in H}))
    say("share of gauged reaches with observed-flow held-out r (iso) below 0.5: "
        f"{(G['obs__iso'] < 0.5).mean():.2f}; by basin {G.groupby('basin')['obs__iso'].apply(lambda x: round((x < 0.5).mean(), 2)).to_dict()}")
    snr = np.log(G.wse_range / G.wse_u)
    say(f"what predicts a gauged reach's ceiling (Spearman with obs iso r): ln(WSE range / wse_u) "
        f"{snr.corr(G['obs__iso'], method='spearman'):+.2f}; ln area {np.log(G.uparea).corr(G['obs__iso'], method='spearman'):+.2f}; "
        f"model-flow iso r {G[f'{RUN}__iso'].corr(G['obs__iso'], method='spearman'):+.2f}")
snr = np.log(R.wse_range / R.wse_u)
say(f"\nall reaches: Spearman of the {RUN} held-out r (iso) with ln(WSE range / wse_u) {snr.corr(R[f'{RUN}__iso'], method='spearman'):+.2f}, "
    f"with ln area {np.log(R.uparea).corr(R[f'{RUN}__iso'], method='spearman'):+.2f}; WSE range / wse_u median {(R.wse_range / R.wse_u).median():.1f}, "
    f"share < 5: {((R.wse_range / R.wse_u) < 5).mean():.2f}")
(DIAG / "swot_head_cv.txt").write_text("\n".join(lines) + "\n")
