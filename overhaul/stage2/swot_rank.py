"""SWOT river rank diagnostic (2026-11-12; no training; user go "1 and 2"): does the modelled flow rise and fall with
the SWOT water-surface elevation of the same reach? Level is monotone in discharge at a reach, so the rank agreement
needs no rating curve and uses elevation only (widths are not read).

Inputs: swot_era_flows.py output (--diag: flows_<basin>.npz, model flow per run and mode at the nodes of the matched
SWORD reaches and at the gauges, 2023-07-01 .. end of record) and data/overhaul/swot_rivers/{reach_match,reach_obs}.
Per reach with >= MIN_DAYS observed days (one value per day: mean WSE of the day's passes) and per run / mode:
  rho      Spearman rank correlation of model flow (reach node, pass day) and WSE;
  rho_jas  the same on Jul-Sep days only (>= 8 days);
  sgn      share of consecutive observed days (<= 45 days apart, |dWSE| >= DWSE m) on which the model flow changes in
           the direction of the WSE.
Check of the signal itself, at reaches that share their unit catchment with a gauge on the same river (gauge MERIT area
within [0.8, 1.02] of the unit's outlet area): rho of the OBSERVED flow with WSE (the ceiling), rho of the model with
the observed flow on the same days, and whether the between-run differences of the WSE score track the differences
of the gauge score (if they do, the WSE rank score is a usable stand-in for flow skill on ungauged reaches).
Gauge observations of 2023-07+ were never used in training or selection (EVAL_SPEC); here they only score.
Statistics are over nodes (reaches of one node averaged), with a node bootstrap (1000 resamples) for paired changes.
Primary set: plain river reaches (SWORD type 1, lakeflag 0); lake / dam reaches are reported apart.
Writes --diag/swot_rank_reaches.parquet, swot_rank_gauges.parquet, swot_rank.txt.
Usage: python swot_rank.py --diag=DIR --control=c2wva,c2wva_s1 --arms=I:I,I_s1;IF:IF,IF_s1;T:T,T_s1 [--base=b]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
DIAG = Path(opt["diag"])
OV = Path("/project/swot-ml/data/overhaul")
CONTROL = opt.get("control", "c2wva,c2wva_s1").split(",")
ARMS = {a.split(":")[0]: a.split(":")[1].split(",") for a in opt.get("arms", "").split(";") if a}
BASE = opt.get("base", "b")
MIN_DAYS, DWSE, MAX_GAP = 10, 0.15, 45
BNAME = {"USGS-07289000": "Mississippi", "USGS-09521100": "Colorado", "USGS-14246900": "Columbia",
         "USGS-02359170": "ACF", "USGS-01578310": "Susquehanna"}
rng = np.random.default_rng(0)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def spearman(x, y):
    if len(x) < 3:
        return np.nan
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    sx, sy = rx.std(), ry.std()
    return float(np.corrcoef(rx, ry)[0, 1]) if sx > 0 and sy > 0 else np.nan


def boot(d, n=1000):
    """Mean and 95% CI of the values in d (one per node)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return np.nan, np.nan, np.nan, len(d)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(d)


RM = pd.read_parquet(OV / "swot_rivers/reach_match.parquet")
RM = RM[(RM.why == "ok") & (RM.n_days >= MIN_DAYS)].set_index("reach_id")
OBS = pd.read_parquet(OV / "swot_rivers/reach_obs.parquet", columns=["reach_id", "date", "wse"])
OBS = OBS.groupby(["reach_id", "date"], as_index=False).wse.mean()
GG = pd.read_parquet(OV / "graph/gauges.parquet").drop_duplicates("gauge")

R_rows, G_rows = [], []
keys = None
for basin, bn in BNAME.items():
    f = DIAG / f"flows_{basin}.npz"
    if not f.exists():
        say(f"missing {f}")
        continue
    z = np.load(f, allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    dpos = pd.Series(np.arange(len(dates)), index=dates)
    nodes, isg = z["nodes"].astype(str), z["is_gauge"]
    npos = {n: i for i, n in enumerate(nodes) if not isg[i]}
    gpos = {n: i for i, n in enumerate(nodes) if isg[i]}  # column in the flow arrays
    gcol = {n: j for j, n in enumerate(nodes[isg])}  # column in obs
    gset = dict(zip(nodes, z["sets"].astype(str)))
    keys = [k for k in z.files if "__" in k]
    flows = {k: z[k] for k in keys}
    qobs = z["obs"]
    rm = RM[RM.basin == basin]
    ob = OBS[OBS.reach_id.isin(rm.index) & OBS.date.isin(dates)]
    # gauges in the same unit catchment and on the same river as a reach
    gg = GG[GG.gauge.isin(gpos) | GG.node.isin(gpos)]
    g_by_comid = {}
    for g, c, a, nd in zip(gg.gauge, gg.comid, gg.area_merit, gg.node):
        nm = g if g in gpos else nd
        g_by_comid.setdefault(int(c), []).append((nm, a))
    for rid, o in ob.groupby("reach_id"):
        r = rm.loc[rid]
        if r.node not in npos or len(o) < MIN_DAYS:
            continue
        ti = dpos.reindex(o.date).values.astype(int)
        w = o.wse.values
        jas = o.date.dt.month.isin([7, 8, 9]).values
        gap = np.diff(o.date.values).astype("timedelta64[D]").astype(int)
        dw = np.diff(w)
        pair = (gap <= MAX_GAP) & (np.abs(dw) >= DWSE)
        row = {"basin": bn, "reach_id": rid, "node": r.node, "comid": int(r.comid), "uparea": r.merit_uparea, "type": r.type, "lakeflag": r.lakeflag,
               "area_ok": bool(r.area_ok), "n": len(o), "n_jas": int(jas.sum()), "n_pair": int(pair.sum()),
               "wse_range": np.percentile(w, 95) - np.percentile(w, 5)}
        j = npos[r.node]
        for k in keys:
            q = flows[k][ti, j]
            row[f"rho__{k}"] = spearman(q, w)
            row[f"rhojas__{k}"] = spearman(q[jas], w[jas]) if jas.sum() >= 8 else np.nan
            row[f"sgn__{k}"] = float((np.sign(np.diff(q)[pair]) == np.sign(dw[pair])).mean()) if pair.sum() >= 5 else np.nan
        ck = f"{CONTROL[0]}__full"
        if ck in flows and f"{CONTROL[0]}__noloss" in flows:
            row["dep_ctl"] = 1 - flows[ck][:, j].mean() / max(flows[f"{CONTROL[0]}__noloss"][:, j].mean(), 1e-9)
        R_rows.append(row)
        for gname, ga in g_by_comid.get(int(r.comid), []):
            if not (0.8 <= ga / r.merit_uparea <= 1.02):
                continue
            qo = qobs[ti, gcol[gname]]
            ok = np.isfinite(qo)
            if ok.sum() < MIN_DAYS:
                continue
            grow = {"basin": bn, "reach_id": rid, "gauge": gname, "set": gset[gname], "uparea": r.merit_uparea, "type": r.type,
                    "lakeflag": r.lakeflag, "n": int(ok.sum()), "rho_obs_wse": spearman(qo[ok], w[ok])}
            jg = gpos[gname]
            for k in keys:
                qm = flows[k][ti, jg][ok]
                grow[f"rho_wse__{k}"] = spearman(qm, w[ok])
                grow[f"rho_q__{k}"] = spearman(qm, qo[ok])
            G_rows.append(grow)

R = pd.DataFrame(R_rows)
Gd = pd.DataFrame(G_rows)
R.to_parquet(DIAG / "swot_rank_reaches.parquet", index=False)
Gd.to_parquet(DIAG / "swot_rank_gauges.parquet", index=False)
R["river"] = (R.type == 1) & (R.lakeflag == 0)
R["size"] = pd.cut(R.uparea, [0, 1e4, 1e5, 1e7], labels=["<10k", "10k-100k", ">=100k"])
runs = sorted({k.split("__")[0] for k in keys})
loss_runs = [r for r in runs if f"{r}__noloss" in keys]
say(f"SWOT river rank diagnostic; runs {runs}; reaches scored {len(R)} on {R.node.nunique()} nodes "
    f"(plain river {int(R.river.sum())}); median observed days {R.n.median():.0f}, Jul-Sep {R.n_jas.median():.0f}, "
    f"direction pairs {R.n_pair.median():.0f}; median 5-95% WSE range {R.wse_range.median():.2f} m")


def node_mean(df, cols):
    return df.groupby(["basin", "node"])[cols].mean()


def arm_cols(stat, names, mode="full"):
    return [f"{stat}__{n}__{mode}" for n in names if f"{stat}__{n}__{mode}" in R.columns]


P = R[R.river]
say("\n## 1. Plain river reaches: median over nodes of rho(model flow, WSE), per run (full)")
t = node_mean(P, [f"rho__{r}__full" for r in runs])
t.columns = runs
tab = t.groupby("basin").median().T
tab["all"] = t.median()
say(tab.round(3).to_string())
say("\nby drainage area (km2), all basins:")
sz = P.groupby(["basin", "node"])["size"].first()
tab = t.groupby(sz.reindex(t.index).values).median().T
say(tab.round(3).to_string())
say("\nJul-Sep only (rho_jas) and direction of change (sgn), median over nodes, all basins:")
tj = node_mean(P, [f"rhojas__{r}__full" for r in runs]).median()
ts = node_mean(P, [f"sgn__{r}__full" for r in runs]).median()
say(pd.DataFrame({"rho_jas": tj.values, "sgn": ts.values}, index=runs).round(3).to_string())


def paired(df, a_cols, b_cols, stat_label):
    """Mean over nodes of (mean of a_cols - mean of b_cols), with CI; by basin and all."""
    nm = node_mean(df, a_cols + b_cols)
    dlt = nm[a_cols].mean(1) - nm[b_cols].mean(1)
    out = []
    for bn, g in list(dlt.groupby("basin")) + [("all", dlt)]:
        m, lo, hi, n = boot(g.values)
        out.append(f"{bn} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (n {n}, {np.mean(g.dropna() > 0):.0%} up)")
    say(f"  {stat_label}: " + "; ".join(out))


say("\n## 2. Paired changes on plain river reaches (mean over nodes, 95% node-bootstrap CI)")
say(f"(a) the losses of a run: full - noloss (same routing, reservoirs and gains)")
for nm_, names in [("control", CONTROL)] + list(ARMS.items()):
    names = [n for n in names if n in loss_runs]
    if not names:
        continue
    for stat in ("rho", "rhojas", "sgn"):
        paired(P, arm_cols(stat, names), arm_cols(stat, names, "noloss"), f"{nm_:8s} {stat:7s}")
say(f"(b) arms against the control (full runs, seed means)")
for nm_, names in ARMS.items():
    for stat in ("rho", "rhojas", "sgn"):
        if arm_cols(stat, names):
            paired(P, arm_cols(stat, names), arm_cols(stat, CONTROL), f"{nm_:8s} {stat:7s}")
if f"rho__{BASE}__full" in R.columns:
    say(f"(c) control against routing only ({BASE})")
    for stat in ("rho", "rhojas", "sgn"):
        paired(P, arm_cols(stat, CONTROL), [f"{stat}__{BASE}__full"], f"control  {stat:7s}")
if "dep_ctl" in P.columns:
    say("(d) as (a), control, by the control's modelled depletion at the node (1 - full / noloss):")
    for lab, m in (("depletion >= 10%", P.dep_ctl >= 0.10), ("2-10%", (P.dep_ctl >= 0.02) & (P.dep_ctl < 0.10)), ("< 2%", P.dep_ctl < 0.02)):
        for stat in ("rho", "rhojas"):
            paired(P[m], arm_cols(stat, CONTROL), arm_cols(stat, CONTROL, "noloss"), f"{lab:16s} {stat:7s}")

HOLD = int(opt.get("hold", 5))  # train_s2 --rsw_hold: reaches with comid % HOLD == 0 never enter the river term
if ARMS and HOLD > 0:
    say(f"\n## 6. Arms against the control on reaches held out of the river term (comid % {HOLD} == 0) and on the others")
    P0 = P
    for lab, m in (("held out", P0.comid % HOLD == 0), ("in the term", P0.comid % HOLD != 0)):
        say(f"{lab}: {int(m.sum())} reaches")
        for nm_, names in ARMS.items():
            for stat in ("rho", "rhojas"):
                if arm_cols(stat, names):
                    paired(P0[m], arm_cols(stat, names), arm_cols(stat, CONTROL), f"{nm_:8s} {stat:7s}")

say("\n## 3. Is the WSE rank score a stand-in for flow skill? (reaches sharing a unit catchment with a gauge)")
if len(Gd):
    Gp = Gd[(Gd.type == 1) & (Gd.lakeflag == 0)]
    say(f"reach-gauge pairs {len(Gd)} (plain river {len(Gp)}, gauges {Gp.gauge.nunique()}); observed days median {Gp.n.median():.0f}")
    say("ceiling: rho(OBSERVED flow, WSE), plain river pairs: median "
        f"{Gp.rho_obs_wse.median():.3f}, p10 {Gp.rho_obs_wse.quantile(.1):.3f}, p25 {Gp.rho_obs_wse.quantile(.25):.3f}, "
        f"share >= 0.8: {(Gp.rho_obs_wse >= 0.8).mean():.2f}; by basin {Gp.groupby('basin').rho_obs_wse.median().round(3).to_dict()}")
    say("by drainage area: " + str(Gp.groupby(pd.cut(Gp.uparea, [0, 1e4, 1e5, 1e7], labels=['<10k', '10k-100k', '>=100k']), observed=False)
                                   .rho_obs_wse.agg(['median', 'size']).round(3).to_dict('index')))
    say("median over pairs, per run (full): rho(model, WSE) and rho(model, observed flow) on the same days")
    say(pd.DataFrame({"rho_wse": [Gp[f"rho_wse__{r}__full"].median() for r in runs],
                      "rho_q": [Gp[f"rho_q__{r}__full"].median() for r in runs]}, index=runs).round(3).to_string())
    say("across pairs, level: Spearman of rho(model, WSE) with rho(model, observed flow), per run: "
        + str({r: round(spearman(Gp[f"rho_wse__{r}__full"].values, Gp[f"rho_q__{r}__full"].values), 2) for r in runs}))
    say("across pairs, CHANGES (does a gain in the WSE score mean a gain in the gauge score?): Spearman / sign agreement")
    cmp = [(f"{r} full - noloss", f"{r}__full", f"{r}__noloss") for r in loss_runs]
    cmp += [(f"{r} - {BASE}", f"{r}__full", f"{BASE}__full") for r in runs if r != BASE and f"rho_q__{BASE}__full" in Gp.columns]
    for lab, a, b in cmp:
        dw, dq = Gp[f"rho_wse__{a}"] - Gp[f"rho_wse__{b}"], Gp[f"rho_q__{a}"] - Gp[f"rho_q__{b}"]
        ok = np.isfinite(dw) & np.isfinite(dq) & (dq.abs() >= 0.02)
        say(f"  {lab:24s} all pairs rho {spearman(dw.values, dq.values):+.2f}; pairs with |change in gauge score| >= 0.02 "
            f"(n {int(ok.sum())}): rho {spearman(dw[ok].values, dq[ok].values):+.2f}, same sign {np.mean(np.sign(dw[ok]) == np.sign(dq[ok])):.0%}")
else:
    say("no reach-gauge pairs")

say("\n## 5. Gauge-to-gauge reaches: does SWOT side with the observations on where water is lost?")
# REACH_TABLES_FOR_SWOT.md section 1: per gauge g, observed and modelled flow change over the reach above g (to the next
# gauges upstream), 1980-2022, control seed 0. Each SWOT reach is put in the reach of the first such gauge at or
# downstream of its node. Classes by the change as a share of the flow at g (loss = change <= -2%):
#   hit = observed loss, model loss; miss = observed loss, no model loss; false = model loss, observations do not lose;
#   none = neither. If SWOT carries the loss signal, switching the model's losses on should raise the rank score in
#   "hit" reaches, do little in "miss" (nothing to switch) and lower it in "false" ones.
RT = Path(opt.get("reach_table", "/project/swot-ml/swot-ml/overhaul/stage2/diag/reachinc_c2wva_f0/gauges.parquet"))
if RT.exists():
    import json
    T = pd.read_parquet(RT)
    T = T[(T.run == CONTROL[0]) & T.inc_obs.notna() & (T.tf_omean > 0)].drop_duplicates("gauge").set_index("gauge")
    T["obs_ch"], T["mod_ch"] = T.inc_obs / T.tf_omean, T.inc_mod / T.tf_omean
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

    P = P.assign(rgauge=[reach_gauge(n) for n in P.node])
    P = P.join(T[["obs_ch", "mod_ch", "up_share"]], on="rgauge")
    P["cls"] = np.select([(P.obs_ch <= -0.02) & (P.mod_ch <= -0.02), (P.obs_ch <= -0.02) & (P.mod_ch > -0.02),
                          (P.mod_ch <= -0.02) & (P.obs_ch > 0)], ["hit", "miss", "false"], default="none")
    P.loc[P.rgauge.isna(), "cls"] = "no gauged reach"
    say(f"plain river SWOT reaches inside a gauged reach: {int(P.rgauge.notna().sum())} of {len(P)}, in {P.rgauge.nunique()} gauged reaches; "
        f"SWOT reaches / gauged reaches by class: "
        + str({c: (int((P.cls == c).sum()), int(P.rgauge[P.cls == c].nunique())) for c in ['hit', 'miss', 'false', 'none']}))
    say("by basin: " + str(P[P.rgauge.notna()].groupby(["basin", "cls"]).size().unstack(fill_value=0).to_dict("index")))
    say("control, full - noloss, by class (mean over nodes, node-bootstrap CI; 'all' = all basins):")
    for c in ("hit", "miss", "false", "none"):
        if (P.cls == c).sum() >= 5:
            for stat in ("rho", "rhojas", "sgn"):
                paired(P[P.cls == c], arm_cols(stat, CONTROL), arm_cols(stat, CONTROL, "noloss"), f"{c:6s} {stat:7s}")
    say("level of the rank score by class, median over nodes (control full | control noloss | routing only):")
    for c in ("hit", "miss", "false", "none"):
        m = P.cls == c
        if m.sum() >= 5:
            nmn = node_mean(P[m], arm_cols("rho", CONTROL) + arm_cols("rho", CONTROL, "noloss") + [f"rho__{BASE}__full"])
            say(f"  {c:6s} {nmn[arm_cols('rho', CONTROL)].mean(1).median():.3f} | {nmn[arm_cols('rho', CONTROL, 'noloss')].mean(1).median():.3f} | "
                f"{nmn[f'rho__{BASE}__full'].median():.3f}  (nodes {len(nmn)})")
    for nm_, names in ARMS.items():
        names = [n for n in names if n in loss_runs]
        if names:
            say(f"arm {nm_}, full - noloss, by class:")
            for c in ("hit", "miss", "false"):
                if (P.cls == c).sum() >= 5:
                    paired(P[P.cls == c], arm_cols("rho", names), arm_cols("rho", names, "noloss"), f"{c:6s} rho    ")
    P.drop(columns=["size"]).to_parquet(DIAG / "swot_rank_reaches_classed.parquet", index=False)
else:
    say(f"reach table {RT} not found")

say("\n## 4. Lake / dam / unreliable reaches (not in the primary set): median rho over reaches, control full")
ck = f"rho__{CONTROL[0]}__full"
say(str(R[~R.river].groupby("type")[ck].agg(["median", "size"]).round(3).to_dict("index")) + " (type 3 lake on river, 4 dam, 5 unreliable, 6 ghost)")
say(f"sensitivity, plain river reaches whose SWORD facc agrees with MERIT within 25% ({int(P.area_ok.sum())} of {len(P)}): "
    f"median rho control {P[P.area_ok][ck].median():.3f} vs others {P[~P.area_ok][ck].median():.3f}")
(DIAG / "swot_rank.txt").write_text("\n".join(lines) + "\n")
