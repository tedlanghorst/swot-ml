"""SWOT river reaches on the overhaul graph (2026-11-12; user go: "1 and 2" = match SWOT river reaches to the MERIT
graph and count usable reaches / passes; then the no-training rank diagnostic, stage2/swot_rank.py).

1. Match (v2, same day: SWORD v17c facc is 1.5-2.3 x the MERIT drainage area on the large main stems, e.g. Columbia
   mouth 1.27M vs 0.67M km2, so an area test rejects them). Geometry only: NPTS points along each SWORD reach
   centreline (regions 73 / 74 / 77 / 78), each takes its nearest MERIT-Basins river line of the five target basins
   within SEARCH_KM; the reach takes the line that most points chose (>= 5 of 9; ties: smaller median distance).
   A tributary reach ending at a confluence keeps its own line because most of its points are nearer to it.
   ln_mis (signed ln distance of facc from the line's [inlet, outlet] area range) and area_ok (within 25%) are kept
   as flags for sensitivity checks, not used to accept.
   The dataset node is the unit catchment's own node (flow at the catchment outlet; a gauge piece inside the same
   unit catchment is ignored: on SWOT-visible rivers the two differ by a part of one ~45 km2 unit's runoff).
2. Observations. RiverSP vD reach rows of the matched reaches with the settled filter (swot-nde build_obs.py; never a
   dschg_* column): observed pass, 0 <= reach_q_b < 32, wse / wse_u / wse_r_u / slope / slope_u / width present,
   no calval cycle (>= 400), one row per (reach, cycle, pass). No outlier filter.
Writes /project/swot-ml/data/overhaul/swot_rivers/: reach_match.parquet (one row per SWORD reach of the regions,
matched or not, with the reason), reach_obs.parquet, summary.txt.
Usage: python swot_reach_match.py   (CPU, a few minutes; sbatch swot_reach_match.sbatch)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyogrio
import shapely
from shapely import STRtree

DATA = Path("/project/swot-ml/data")
OV = DATA / "overhaul"
OUT = OV / "swot_rivers"
OUT.mkdir(parents=True, exist_ok=True)
SEARCH_KM, MAX_LN, NPTS = 1.0, float(np.log(1.25)), 9
REGIONS = {73: ["USGS-02359170", "USGS-01578310"], 74: ["USGS-07289000"], 77: ["USGS-09521100"], 78: ["USGS-14246900"]}
BNAME = {"USGS-07289000": "Mississippi", "USGS-09521100": "Colorado", "USGS-14246900": "Columbia",
         "USGS-02359170": "ACF", "USGS-01578310": "Susquehanna"}
REQUIRED = ["wse", "wse_u", "wse_r_u", "slope", "slope_u", "width"]
COLS = ["reach_id", "time", "time_str", "wse", "wse_u", "wse_r_u", "slope", "slope_u", "width", "width_u", "reach_q",
        "reach_q_b", "dark_frac", "ice_clim_f", "partial_f", "n_good_nod", "obs_frac_n", "xovr_cal_q", "xtrk_dist",
        "cycle_id", "pass_id", "version"]
assert not any(c.startswith("dschg") for c in COLS)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


# dataset nodes of the five basins and their unit catchments
z = np.load(OV / "ds_targets/segments_tau1.npz", allow_pickle=True)
ds_names, ds_basin = z["names"].astype(str), z["basin"].astype(str)
N = pd.read_parquet(OV / "graph/nodes.parquet")
G = pd.read_parquet(OV / "graph/gauges.parquet").drop_duplicates("gauge")
g2ds = dict(zip(G.node, G.gauge))  # graph node -> gauge name (dataset nodes at gauges carry the gauge name)
in_ds = dict(zip(ds_names, ds_basin))
units = N[N.kind == "unit"]
rows = []
for gnode, comid in zip(units.index, units.comid):
    for cand in (gnode, g2ds.get(gnode)):
        if cand in in_ds:
            rows.append((int(comid), cand, in_ds[cand]))
            break
U = pd.DataFrame(rows, columns=["comid", "node", "basin"]).drop_duplicates("comid").set_index("comid")
say(f"dataset nodes {len(ds_names)}; unit catchments with a dataset node {len(U)}")

S = pq.read_table(DATA / "sword/parquet/sword_NA_v17c_reaches.parquet",
                  columns=["reach_id", "x", "y", "facc", "reach_length", "width", "lakeflag", "type", "n_chan_max",
                           "obstr_type", "dist_out", "river_name", "geometry"]).to_pandas()
S["region"] = S.reach_id // 10 ** 9
S = S[S.region.isin(list(REGIONS))].reset_index(drop=True)
say(f"SWORD v17c reaches in regions {sorted(REGIONS)}: {len(S)}")

out = []
for reg in REGIONS:
    R = pyogrio.read_dataframe(DATA / f"merit_mghydro_v3/rivers{reg}.db", layer="rivers",
                               columns=["comid", "uparea", "up1", "up2", "up3", "up4"], fid_as_index=False)
    if "comid" not in R.columns:  # comid is the primary key (fid)
        R = pyogrio.read_dataframe(DATA / f"merit_mghydro_v3/rivers{reg}.db", layer="rivers",
                                   columns=["uparea", "up1", "up2", "up3", "up4"], fid_as_index=True)
        R["comid"] = R.index.astype("int64")
    R = R.reset_index(drop=True)
    ua = dict(zip(R.comid, R.uparea))
    R["inlet"] = sum(R[c].map(ua).fillna(0.0) for c in ["up1", "up2", "up3", "up4"])
    R = R[R.comid.isin(U.index)].reset_index(drop=True)
    tree = STRtree(R.geometry.values)
    s = S[S.region == reg].reset_index(drop=True)
    geom = shapely.from_wkb(s.geometry.values)
    kmdeg = 111.0 * np.cos(np.radians(s.y.values))
    kms = np.sqrt((111.0 ** 2 + kmdeg ** 2) / 2)  # km per degree, mean of the two axis scales
    # nearest MERIT line of NPTS points along the reach; the reach takes the line most of its points are nearest to
    fr = (np.arange(NPTS) + 0.5) / NPTS
    pts = shapely.line_interpolate_point(geom[:, None], fr[None, :], normalized=True)  # (n, NPTS)
    qi, ri = tree.query_nearest(pts.ravel(), max_distance=SEARCH_KM / 80.0, all_matches=False)
    C = pd.DataFrame({"q": qi // NPTS, "r": ri})
    C["dist_km"] = shapely.distance(pts.ravel()[qi], R.geometry.values[ri]) * kms[C.q.values]
    C = C[C.dist_km <= SEARCH_KM]
    V = C.groupby(["q", "r"]).agg(votes=("dist_km", "size"), dist_km=("dist_km", "median")).reset_index()
    best = V.sort_values(["votes", "dist_km"], ascending=[False, True]).drop_duplicates("q").set_index("q")
    s["comid"] = pd.Series(R.comid.values[best.r.values], index=best.index).reindex(s.index).astype("Int64")
    s["votes"] = best.votes.reindex(s.index)
    s["dist_km"] = best.dist_km.reindex(s.index)
    s["merit_uparea"] = pd.Series(R.uparea.values[best.r.values], index=best.index).reindex(s.index)
    s["merit_inlet"] = pd.Series(R.inlet.values[best.r.values], index=best.index).reindex(s.index)
    lo, hi = np.maximum(s.merit_inlet, 1e-3), s.merit_uparea
    s["ln_mis"] = np.where(s.facc < lo, -np.log(lo / np.maximum(s.facc, 1e-3)), np.where(s.facc > hi, np.log(s.facc / hi), 0.0))
    s["area_ok"] = s.ln_mis.abs() <= MAX_LN
    s["why"] = np.where(s.comid.isna(), "no MERIT line of the target basins within 1 km",
                        np.where(s.votes < NPTS // 2 + 1, "no majority line", "ok"))
    s = s.drop(columns="geometry")
    ok = s.why == "ok"
    s["node"] = None
    s["basin"] = None
    s.loc[ok, "node"] = U.node.reindex(s.comid[ok].astype("int64")).values
    s.loc[ok, "basin"] = U.basin.reindex(s.comid[ok].astype("int64")).values
    out.append(s)
    say(f"region {reg}: SWORD reaches {len(s)}, matched {int(ok.sum())} (area within 25%: {int((ok & s.area_ok).sum())}); "
        f"reasons {s.why.value_counts().to_dict()}")
M = pd.concat(out, ignore_index=True)
M["basin_name"] = M.basin.map(BNAME)

# observations of the matched reaches
ids = M.reach_id[M.why == "ok"].astype(str).tolist()
d = ds.dataset(DATA / "swot/RiverSP_vD_20260812/parquet/NA.parquet")
t = d.to_table(filter=ds.field("reach_id").isin(ids), columns=COLS).to_pandas()
assert not any(c.startswith("dschg") for c in t.columns)
n0 = len(t)
for c in t.columns:
    if t[c].dtype.kind == "f":
        t[c] = t[c].where(t[c] > -1e11, np.nan)
t["reach_id"] = t.reach_id.astype("int64")
t["cycle"], t["pass"] = t.cycle_id.astype(int), t.pass_id.astype(int)
drops = []


def drop(mask, why):
    global t
    drops.append((why, int(mask.sum())))
    t = t[~mask]


drop(t.time.isna() | (t.time_str == "no_data"), "no observation in the pass")
drop(~((t.reach_q_b >= 0) & (t.reach_q_b < 32)), "reach_q_b >= 32")
drop(t[REQUIRED].isna().any(axis=1), "required field missing")
drop(t.cycle >= 400, "calval cycle")
dup = t.sort_values("version").duplicated(["reach_id", "cycle", "pass"], keep="first")
drop(dup.reindex(t.index), "duplicate (reach, cycle, pass)")
t["time_utc"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(t.time, unit="s")
t["date"] = t.time_utc.dt.floor("D")
t = t.drop(columns=["cycle_id", "pass_id", "time", "time_str"]).sort_values(["reach_id", "time_utc"]).reset_index(drop=True)
say(f"RiverSP rows of matched reaches {n0}; kept {len(t)}; drops {drops}")
say(f"time span {t.time_utc.min():%Y-%m-%d} .. {t.time_utc.max():%Y-%m-%d}")

per = t.groupby("reach_id").agg(n_obs=("date", "size"), n_days=("date", "nunique"), wse_range=("wse", lambda x: x.quantile(.95) - x.quantile(.05)),
                                wse_u_med=("wse_u", "median"))
M = M.merge(per, left_on="reach_id", right_index=True, how="left")
M["n_obs"] = M.n_obs.fillna(0).astype(int)
M["n_days"] = M.n_days.fillna(0).astype(int)
M.to_parquet(OUT / "reach_match.parquet", index=False)
t.to_parquet(OUT / "reach_obs.parquet", index=False)

ok = M[M.why == "ok"].copy()
ok["river"] = (ok.type == 1) & (ok.lakeflag == 0)  # plain river reaches (no lake, dam, unreliable or ghost reach)
ok["size"] = pd.cut(ok.merit_uparea, [0, 1e3, 1e4, 1e5, 1e7], labels=["<1k", "1k-10k", "10k-100k", ">=100k"])
say("\nmatched reaches by basin (all types): reaches, distinct nodes, reaches with >= 10 / >= 30 obs days, median obs days (observed reaches)")
for bn, g in ok.groupby("basin_name"):
    say(f"  {bn:12s} {len(g):6d} {g.node.nunique():6d} {int((g.n_days >= 10).sum()):6d} {int((g.n_days >= 30).sum()):6d} "
        f"{g.n_days[g.n_days > 0].median():6.0f}")
say("\nplain river reaches (type 1, lakeflag 0) with >= 10 obs days, by basin and drainage area (km2): reaches / distinct nodes")
u = ok[ok.river & (ok.n_days >= 10)]
tab = u.groupby(["basin_name", "size"], observed=False).agg(reaches=("reach_id", "size"), nodes=("node", "nunique"))
say(tab.unstack("size").to_string())
say(f"\nplain river reaches >= 10 obs days: {len(u)} on {u.node.nunique()} nodes; median obs days {u.n_days.median():.0f}; "
    f"median 5-95% WSE range {u.wse_range.median():.2f} m; median wse_u {u.wse_u_med.median():.3f} m")
say(f"match quality (matched): median distance {ok.dist_km.median():.2f} km, p95 {ok.dist_km.quantile(.95):.2f}; "
    f"SWORD facc within 25% of the MERIT range {ok.area_ok.mean():.2f}; median facc / MERIT area by size "
    f"{ok.assign(r=ok.facc / ok.merit_uparea).groupby('size', observed=False).r.median().round(2).to_dict()}")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
