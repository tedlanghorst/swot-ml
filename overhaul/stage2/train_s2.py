"""Stage-2 training (2026-10-14, overhaul): R2 routing (+ losses, gains, reservoirs) on frozen stage-1 runoff.

Windows: 1,092 days (156 weeks): 364 days of spin-up, then 728 scored days, starting on a week boundary, inside the
training period (to 2022-12-31). One basin per step, sampled in proportion to its train gauges.
Loss: mean over train gauges of mean squared error / (gauge sd over the training period + sd_eps)² on scored days
with observations (NSE*-style: sd_eps bounds the weight of near-constant gauges, whose 1 - NSE is unbounded),
plus lam_loss * mean(1 - pass-through) (L2: area-weighted mean demand share, a volume) and lam_gain * mean(gain / mean local volume), plus (--lam_vol > 0,
2026-10-21) lam_vol * mean over train gauges and 91-day blocks of the squared log ratio of mean flow (offset 0.1 x
gauge sd; blocks with < 50% of days observed left out), as stage 1's volume_loss but per season.
Loss form (--loss_form): l1 = proportional pass-through per node (L1); l2 = capped absolute withdrawal (s2_core.l2_pass:
demand in m3/s from the loss net x area x mean PET, needs a first routing pass without withdrawals).
Reservoir evaporation (--res_evap=1, 2026-10-23, with --res=1): E = GDW area x (S / cap)^(2/3) x daily PET at the dam
node (s2_core.reservoir_scan; no free parameter; --res_area=poly: GDW SWBD polygon area, the first runs; max: the
larger of polygon and GDW reported area AREA_REP, 2026-10-23 late, Lake Powell 121 -> 651 km²), leaving the pool before the release (synthetic_res_balance: E
recovered within ~2-10%). Pool withdrawals are not modelled (the pool vs river-above-dam split is not identifiable).
Reservoir options (2026-10-23, option C: let storage do storage's job): --dam_min=MCM (default 100) lowers the dam
size threshold; --dam_emb=k gives each dam k free policy inputs (a learned per-dam embedding, zero at start, penalty
--lam_emb x mean square), so dams with similar attributes can still operate differently (Mead vs Powell).
--dsd=1 (needs --loss=1 --loss_form=l2 --res=1): two time-varying policy inputs per dam, the L2 withdrawal demand
summed along its downstream path to the next dam and to the outlet (log1p(sum / iref), weekly, gradient stopped;
s2_core.downstream_demand): information flows upstream to the dams, as delivery orders do.
SWOT storage (--swot=1, 2026-10-26, with --res=1; user go): SWOT LakeSP reservoir storage changes as a second training
target (data_prep/lakesp_storage.py; 352 dam nodes, 2023-07..2025-12; Powell and Mead fail the level-fit QC). With
probability --p_swot a step draws a window whose scored days overlap the SWOT period by >= 182 days (basin in proportion
to its observed dams) and its loss is lam_s * mean over observed dams of log1p(NSE-style storage-anomaly error)
(s2_core.swot_storage: model and SWOT storage each about their own mean over the window's observed days; / variance of
the SWOT anomaly + noise) plus the usual penalties. No gauge enters after 2022-12-31 (obs_ok is cut there for every
run), so gauges and SWOT never overlap in time (EVAL_SPEC). Selection unchanged (validation gauges); eval logs the
in-sample SWOT fit (median over dams of the anomaly NSE, last two years of the record).
Carried storage (--carry=1, 2026-11-10, with --res=1, not --swot): validation runs the windows back to back, each
starting its dams at the storage the previous window had on that day (as ladder_diag --spin=cont), and keeps that
1980-2022 trajectory as a per-basin bank (storage and inflow EMAs at the start of every day, from the latest
evaluation; first one at step 0). Every training window then starts its dams from the bank at its first day (gradient
stopped) instead of half full, so dams carry multi-year storage between windows and the policy trains on the fills
it actually reaches. Selection uses the chained validation.
Long memory (--long_mem=1, with --res=1): two more release-policy inputs per dam, log1p of the inflow EMA with 1- and
3-year time constants / iref (s2_core.reservoir_scan lm); with --carry=1 the EMA states come from the bank too.
Demand form (--demand=free|irr|both, 2026-11-12, with --loss_form=l2; s2_core.l2_demand): free = the withdrawal demand
above (softplus x node area x mean PET, anywhere, unbounded); irr = kc x (irrigated + urban land) x weekly PET with
kc in (0, 1.2) from a second net on the same inputs: bounded by crop water demand, zero where there is no such land,
seasonal by construction; both = their sum, the free part penalised --lam_free per m3 (default = --lam_loss), for
transfers and whatever irrigation cannot explain.
Teacher-forced gauge loss (--tf=w, 2026-11-12): the gauge terms become (1 - w) x the usual loss + w x the same loss
on teacher-forced predictions (s2_core.tf_volumes): at each train gauge the routed error of the nearest upstream train
gauges is taken out, so the reach between gauges is fitted to its own balance and upstream runoff error does not
land in its loss. Training only; validation and every diagnostic stay free-running.
Offline release policy (--policy_init=FILE [--policy_freeze=1], 2026-11-12, with --res=1): start the release policy
from policy_resops.py (fitted to ResOpsUS inflow / storage / release; user decision 2026-11-12, ResOpsUS no longer
evaluation only for these runs) and, with freeze, keep it fixed.
Validation: every --eval_every steps, each basin over 1980-2022 in non-overlapping 728-day scored windows (each with
its own 364-day spin-up); NSE per validation gauge over all scored days; selection metric = mean over basins of
the median gauge NSE (EVAL_SPEC).
Outputs in --out: log.jsonl, params_last.eqx, params_best.eqx.

Usage: python train_s2.py --out=DIR --runoff=DIR [--basins=a,b] [--loss=0|1] [--res=0|1] [--minutes=60]
       [--lr=3e-3] [--eval_every=200] [--lam_loss=0.1] [--lam_gain=0.1] [--fold=0] [--seed=0] [--sd_eps=0.0015]
       [--init=params.eqx] [--diag=1] [--folds=lists_topo|lists_topo_qc|lists]
       [--route_init=stage2/route_init_<arm>.npz --route_init_mode=hybrid|stage1]
       [--loss_form=l1|l2 --wcap=0.9] [--lam_vol=0] [--res_evap=0|1 --res_area=poly|max] [--dam_min=100] [--dam_emb=0 --lam_emb=1e-3] [--dsd=0|1]
       [--swot=0|1 --lam_s=1 --p_swot=0.25] [--carry=0|1] [--long_mem=0|1]
       [--demand=free|irr|both --lam_free=<lam_loss>] [--tf=0] [--policy_init=FILE --policy_freeze=0|1]
"""
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

sys.path.insert(0, "/project/swot-ml/swot-ml/overhaul/stage2")
from s2_core import (KM2_MM_TO_M3S, downstream_demand, l2_demand, tf_pairs, tf_volumes, LOSS_STATICS, ROUTE_STATICS, Basin, S2Model, _std, accumulate_r2, l2_pass,  # noqa: E402
                     r2_route_res, swot_storage)

opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
OUT = Path(opt["out"])
OUT.mkdir(parents=True, exist_ok=True)
OV = "/project/swot-ml/data/overhaul"
BASINS = opt.get("basins", "USGS-07289000,USGS-09521100,USGS-14246900,USGS-02359170,USGS-01578310").split(",")
USE_LOSS, USE_RES = opt.get("loss", "0") == "1", opt.get("res", "0") == "1"
RES_EVAP = USE_RES and opt.get("res_evap", "0") == "1"
RES_AREA = opt.get("res_area", "poly")
assert RES_AREA in ("poly", "max") and (RES_EVAP or RES_AREA == "poly"), "--res_area=poly|max, with --res_evap=1"
DAM_MIN, DAM_EMB = float(opt.get("dam_min", 100)), int(opt.get("dam_emb", 0)) if USE_RES else 0
LAM_E = float(opt.get("lam_emb", 1e-3))
LAM_L, LAM_G = float(opt.get("lam_loss", 0.1)), float(opt.get("lam_gain", 0.1))
L2 = opt.get("loss_form", "l1") == "l2"
DSD = USE_RES and USE_LOSS and L2 and opt.get("dsd", "0") == "1"
assert DSD or opt.get("dsd", "0") != "1", "--dsd=1 needs --loss=1 --loss_form=l2 --res=1"
W_CAP = float(opt.get("wcap", 0.9))  # L2: largest share of a node's flow its withdrawal can take
LAM_V = float(opt.get("lam_vol", 0.0))
SWOT = USE_RES and opt.get("swot", "0") == "1"
assert SWOT or opt.get("swot", "0") != "1", "--swot=1 needs --res=1"
LAM_S, P_SW = float(opt.get("lam_s", 1.0)), float(opt.get("p_swot", 0.25))
# 2026-11-12 (worktree swot-river-rank; user go): amplitude-free SWOT terms.
# --swot_form=corr: reservoir term = 1 - correlation of modelled and SWOT storage per dam (cannot be lowered by damping
#   the pools, the failure of the squared-error form in c2wvas); nse = the old form.
# --rsw=LAM: river term on SWOT-era windows: per plain river SWORD reach, 1 - rho, rho = Pearson correlation of
#   sqrt(routed flow) and the SWOT WSE on the reach's observed days in the window (the square-root head with its offset
#   and slope solved out; offline head fit: as good as a free monotone curve on held-out passes). Per reach a learned
#   noise scale s in [rsw_smin, rsw_smax]: loss = (1 - rho) / s + log s. s <= 0.5 = the reach's ceiling is assumed
#   no lower than 0.5 (user: a noise floor, so the model cannot learn to ignore a reach); s >= 0.1 = no higher than 0.9.
#   --rsw_hold=5: reaches whose MERIT comid % 5 == 0 are held out of the term (scored only).
SWOT_FORM = opt.get("swot_form", "nse")
assert SWOT_FORM in ("nse", "corr")
LAM_R = float(opt.get("rsw", 0.0))
RSW = LAM_R > 0
RSW_SMIN, RSW_SMAX, RSW_HOLD = float(opt.get("rsw_smin", 0.1)), float(opt.get("rsw_smax", 0.5)), int(opt.get("rsw_hold", 5))
RSW_MIN_DAYS, RSW_MIN_WIN = 20, 10
CARRY = USE_RES and opt.get("carry", "0") == "1"
LONG = USE_RES and opt.get("long_mem", "0") == "1"
assert (CARRY or opt.get("carry", "0") != "1") and (LONG or opt.get("long_mem", "0") != "1"), "--carry / --long_mem need --res=1"
assert not (CARRY and (SWOT or float(opt.get("rsw", 0)) > 0)), "--carry=1 with --swot=1 / --rsw not implemented (SWOT windows lie past the bank)"
DEMAND = opt.get("demand", "free")
assert DEMAND in ("free", "irr", "both") and (DEMAND == "free" or (USE_LOSS and L2)), "--demand=irr|both needs --loss=1 --loss_form=l2"
LAM_F = float(opt.get("lam_free", LAM_L))
TF_W = float(opt.get("tf", 0.0))
POL_FREEZE = USE_RES and opt.get("policy_freeze", "0") == "1"
assert USE_RES or "policy_init" not in opt, "--policy_init needs --res=1"
SD_EPS = float(opt.get("sd_eps", 0.0015))  # m3/s/km², ~10% of the median gauge sd over the 5 basins
SPIN, SCORED = 364, 728
TW = SPIN + SCORED
rng = np.random.default_rng(int(opt.get("seed", 0)))
json.dump(opt, open(OUT / "config.json", "w"), indent=1)


def log(d):
    d["time"] = time.strftime("%H:%M:%S")
    print(json.dumps(d), flush=True)
    with open(OUT / "log.jsonl", "a") as f:
        f.write(json.dumps(d) + "\n")


t_load = time.time()
B = [Basin(b, opt["runoff"], fold=int(opt.get("fold", 0)), folds=opt.get("folds", "lists_topo"),
           route_init=opt.get("route_init"), route_init_mode=opt.get("route_init_mode", "hybrid"), dam_min_mcm=DAM_MIN)
     for b in BASINS]
# pooled standardisation (all basins), so the nets see the same scale everywhere
ref_r = _std(pd.concat([b.A[ROUTE_STATICS] for b in B]))[1]
ref_l = _std(pd.concat([b.A[LOSS_STATICS] for b in B]))[1]
drv_all = np.concatenate([b.drv.reshape(-1, 2) for b in B[:1]])  # Mississippi-first basins dominate; fine as scale
drv_mu, drv_sd = drv_all.mean(0), drv_all.std(0) + 1e-9
train_end = {b.basin: int(np.searchsorted(b.dates, np.datetime64("2022-12-31"), side="right")) for b in B}
sw_range = {b.basin: (max(0, int(np.searchsorted(b.dates, np.datetime64("2023-07-01"))) - SPIN - SCORED + 182), len(b.dates) - TW)
            for b in B}


def device_data(b):
    st = {k: jnp.asarray(v) for k, v in b.st.items() if isinstance(v, np.ndarray)}
    d = {"st": st, "xs": jnp.asarray(_std(b.A[ROUTE_STATICS], ref_r)[0]), "xl": jnp.asarray(_std(b.A[LOSS_STATICS], ref_l)[0]),
         "tau": jnp.asarray(b.tau), "route_base": {k: jnp.asarray(v) for k, v in b.route_base.items()}, "vol": jnp.asarray(b.vol), "drv": jnp.asarray((b.drv - drv_mu) / drv_sd),
         "obs": jnp.asarray(np.nan_to_num(b.obs, nan=0.0)),
         # 2026-10-26: gauges after 2022 never enter a loss (test window); training windows ended there before anyway
         "obs_ok": jnp.asarray(np.isfinite(b.obs) & (np.arange(len(b.dates)) < train_end[b.basin])[:, None]),
         "obs_sd": jnp.asarray(np.nan_to_num(b.obs_sd, nan=1.0) + SD_EPS), "gauge": jnp.asarray(b.gauges),
         "gup": jnp.asarray(b.up_area[b.gauges]), "vmean": jnp.asarray(b.vmean),
         "doy": jnp.asarray(2 * np.pi * b.dates.dayofyear.values / 365.25, jnp.float32)}
    if L2:
        # withdrawal scale: node area x mean PET over the training weeks (m3/s)
        pet = b.drv[: train_end[b.basin] // 7, :, 0].mean(0)
        d["wscale"] = jnp.asarray((b.area * pet * KM2_MM_TO_M3S).astype(np.float32))
        d["a_use"] = jnp.asarray(b.a_use)
    if TF_W > 0:
        tp = tf_pairs(b)
        b.n_tf = len(tp["u"])
        if b.n_tf:
            d["tf"] = {k: jnp.asarray(v) for k, v in tp.items()}
    if USE_RES and len(b.dam_node):
        d["res"] = {"seg": jnp.asarray(b.dam_seg), "node": jnp.asarray(b.dam_node), "cap": jnp.asarray(b.dam_cap),
                    "iref": jnp.asarray(b.dam_iref), "feat": jnp.asarray(b.dam_feat), "lv_idx": jnp.asarray(b.dam_lv_idx)}
        if RES_EVAP:  # daily PET at the dam nodes (mm/day), sliced per window in forward
            d["res"]["area"] = jnp.asarray(b.dam_area_max if RES_AREA == "max" else b.dam_area)
            d["res_pet"] = jnp.asarray(np.maximum(b.dam_pet, 0.0))
        if DSD:
            d["dsd_pairs"] = {k: jnp.asarray(v) for k, v in b.dsd_pairs.items()}
        if SWOT:
            V_, ok_, den_ = swot_storage(b)
            d["s_obs"], d["s_ok"], d["s_den"] = jnp.asarray(V_), jnp.asarray(ok_.astype(np.float32)), jnp.asarray(den_)
            b.n_sdam = int(ok_.any(0).sum())
    if RSW:
        ro = river_obs(b)
        b.n_rsw, b.n_rsw_train = len(ro["tr"]), int(ro["tr"].sum())
        if b.n_rsw:
            d["rsw"] = {k: jnp.asarray(v) for k, v in ro.items()}
    return d


def river_obs(b):
    """SWOT reach observations of basin b for the river term: plain river reaches (SWORD type 1, lakeflag 0) matched to
    a node of the basin (data_prep/swot_reach_match.py), with >= RSW_MIN_DAYS observed days inside the record from
    2023-07-01; one WSE per reach and day (mean of the day's passes), about the reach mean. t = day index of the
    record, n = node, r = reach index, y = WSE anomaly (m), tr (R,) = reach is in the loss (not held out)."""
    M = pd.read_parquet(f"{OV}/swot_rivers/reach_match.parquet")
    M = M[(M.why == "ok") & (M.type == 1) & (M.lakeflag == 0) & (M.basin == b.basin)]
    idx = {s_: i for i, s_ in enumerate(b.names)}
    M = M[M.node.isin(idx)].set_index("reach_id")
    O = pd.read_parquet(f"{OV}/swot_rivers/reach_obs.parquet", columns=["reach_id", "date", "wse"])
    O = O[O.reach_id.isin(M.index) & (O.date >= "2023-07-01")].groupby(["reach_id", "date"], as_index=False).wse.mean()
    O["t"] = b.dates.get_indexer(pd.DatetimeIndex(O.date))
    O = O[O.t >= 0]
    O = O[O.groupby("reach_id").t.transform("size") >= RSW_MIN_DAYS]
    rid = np.sort(O.reach_id.unique())
    rpos = {r: k for k, r in enumerate(rid)}
    y = O.wse - O.groupby("reach_id").wse.transform("mean")
    hold = (M.comid.reindex(rid).astype("int64").values % RSW_HOLD == 0) if RSW_HOLD > 0 else np.zeros(len(rid), bool)
    return {"t": O.t.values.astype(np.int32), "n": np.array([idx[x] for x in M.node.reindex(O.reach_id).values], np.int32),
            "r": O.reach_id.map(rpos).values.astype(np.int32), "y": y.values.astype(np.float32), "tr": ~hold}


for b in B:
    b.n_sdam = b.n_tf = b.n_rsw = b.n_rsw_train = 0
DD = {b.basin: device_data(b) for b in B}
n_dam_feat = B[0].dam_feat.shape[1]
model = S2Model(len(ROUTE_STATICS), len(LOSS_STATICS), n_dam_feat + DAM_EMB + 2 * DSD + 2 * LONG, USE_LOSS, USE_RES,
                jax.random.PRNGKey(int(opt.get("seed", 0))),  # 2026-10-23: init follows --seed too (was always 0)
                use_irr=DEMAND != "free")
if "policy_init" in opt:
    model = eqx.tree_at(lambda m: m.policy, model, eqx.tree_deserialise_leaves(opt["policy_init"], model.policy))
params = {"model": model, "gain": {b.basin: jnp.full(b.N, -6.0) for b in B} if USE_LOSS else None}
if RSW:  # per-reach noise logit; start at the geometric middle of [smin, smax] (s = 0.224 for 0.1..0.5)
    _s0 = float(np.sqrt(RSW_SMIN * RSW_SMAX))
    params["rsw"] = {b.basin: jnp.full(max(b.n_rsw, 1), float(np.log((_s0 - RSW_SMIN) / (RSW_SMAX - _s0)))) for b in B}
if DAM_EMB:
    params["dam_emb"] = {b.basin: jnp.zeros((max(len(b.dam_node), 1), DAM_EMB)) for b in B}
log({"event": "loaded", "seconds": round(time.time() - t_load), "basins": {b.basin: {"nodes": b.N,
     "train": len(b.g_train), "val": len(b.g_val), "test": len(b.g_test), "dams": len(b.dam_node), "swot_dams": b.n_sdam, "tf_pairs": b.n_tf, "rsw_reaches": b.n_rsw, "rsw_train": b.n_rsw_train}
    for b in B}})


def forward(params, d, n_seg, n_lev, t0, bname, S0=None, E0=None, flux=False, aux=False):
    """Node volumes (TW, N) for the window starting at day t0 (a multiple of 7), the loss penalty (None without
    losses) and dam storage (with flux=True also the dam fluxes, (T, D, 3 [+ 3 EMA states])). S0 (D,) / E0 (D, 3):
    initial dam storage / inflow EMAs (--carry=1). aux=True: (V, pen, S, routing parameters, log pass-through)."""
    m = params["model"]
    policy = m.policy
    if POL_FREEZE:
        pd_, ps_ = eqx.partition(policy, eqx.is_array)
        policy = eqx.combine(jax.lax.stop_gradient(pd_), ps_)
    vol = jax.lax.dynamic_slice_in_dim(d["vol"], t0, TW, 0)
    s = None
    if USE_LOSS:
        drv = jax.lax.dynamic_slice_in_dim(d["drv"], t0 // 7, TW // 7, 0)
        doy_w = jax.lax.dynamic_slice_in_dim(d["doy"], t0, TW, 0)[3::7]
        sc_w = jnp.stack([jnp.sin(doy_w), jnp.cos(doy_w)], -1)
        if L2:  # weekly demand, m3/s, turned into a pass-through below
            d_free, d_irr = l2_demand(m, d["xl"], drv, sc_w, d["wscale"], DEMAND, d["a_use"],
                                      drv[..., 0] * float(drv_sd[0]) + float(drv_mu[0]))
            dem_w = sum(x for x in (d_free, d_irr) if x is not None)
        else:
            s = jnp.repeat(m.loss_field(d["xl"], drv, sc_w), 7, axis=0)  # L1: log pass-through
        gvol = jax.nn.softplus(params["gain"][bname]) * d["vmean"]
        # exact forward; gradient through the upstream sum (no 23 GB cotangent of the runoff array)
        vol = vol + jax.lax.stop_gradient(gvol)[None, :]
    prm = m.route_params(d["xs"], d["tau"], d["route_base"])
    res = None
    if "res" in d:
        doy = jax.lax.dynamic_slice_in_dim(d["doy"], t0, TW, 0)
        res = dict(d["res"], sc=jnp.stack([jnp.sin(doy), jnp.cos(doy)], -1))
        if DAM_EMB:
            res["feat"] = jnp.concatenate([res["feat"], params["dam_emb"][bname][: res["feat"].shape[0]]], -1)
        if "res_pet" in d:
            res["pet"] = jax.lax.dynamic_slice_in_dim(d["res_pet"], t0, TW, 0)
        if DSD:
            res["dsd"] = downstream_demand(dem_w, d["dsd_pairs"], d["res"]["iref"], TW)
        res["long_mem"] = LONG
        if S0 is not None:
            res["S0"] = jax.lax.stop_gradient(S0)
        if E0 is not None:
            res["E0"] = jax.lax.stop_gradient(E0)
    pen = None
    if USE_LOSS and L2:
        # penalty on the withdrawn volume (demand / area x PET, area-weighted): the same cost per m3 at every node.
        # (L1's mean share over nodes makes a m3 cheaper where the flow through the node is large: main stems.)
        # --demand=both: the free part costs lam_free per m3 (penalties() multiplies by lam_loss)
        pen = sum((LAM_F / LAM_L if k == 0 else 1.0) * jnp.sum(x.mean(0)) for k, x in enumerate((d_free, d_irr))
                  if x is not None) / jnp.sum(d["wscale"])
        V0, _ = r2_route_res(vol, prm, d["st"], n_seg, n_lev, res=res, policy=policy)
        s = l2_pass(jnp.repeat(dem_w, 7, axis=0), jax.lax.stop_gradient(V0), W_CAP)
    elif USE_LOSS:
        pen = jnp.mean(1 - jnp.exp(s))
    out = r2_route_res(vol, prm, d["st"], n_seg, n_lev, res=res, policy=policy, s=s, return_flux=flux)
    V, S = out[0], out[1]
    if USE_LOSS:
        acc = accumulate_r2(gvol, d["st"], n_seg, n_lev)
        V = V + (acc - jax.lax.stop_gradient(acc))[None, :]
    if aux:
        return V, pen, S, prm, s
    return (V, pen, S, out[2]) if flux else (V, pen, S)


def loss_fn(params, d, n_seg, n_lev, t0, gmask, bname, S0=None, E0=None):
    V, pen, _, prm, s = forward(params, d, n_seg, n_lev, t0, bname, S0, E0, aux=True)
    Vg = V[:, d["gauge"]]
    o = jax.lax.dynamic_slice_in_dim(d["obs"], t0 + SPIN, SCORED, 0)
    ok = jax.lax.dynamic_slice_in_dim(d["obs_ok"], t0 + SPIN, SCORED, 0) & gmask[None, :]
    L = gauge_loss(Vg[SPIN:] / d["gup"][None, :], o, ok, d)
    if "tf" in d:  # teacher-forced: routed error of the upstream train gauges taken out (whole window, spin-up too)
        o_w = jax.lax.dynamic_slice_in_dim(d["obs"], t0, TW, 0) * d["gup"][None, :]
        ok_w = jax.lax.dynamic_slice_in_dim(d["obs_ok"], t0, TW, 0) & gmask[None, :]
        Vtf = tf_volumes(Vg, o_w, ok_w, d["tf"], prm, s)
        L = (1 - TF_W) * L + TF_W * gauge_loss(Vtf[SPIN:] / d["gup"][None, :], o, ok, d)
    return L + penalties(params, d, pen, bname)


def gauge_loss(q, o, ok, d):
    se = jnp.where(ok, ((q - o) / d["obs_sd"][None, :]) ** 2, 0.0)
    per_g = se.sum(0) / jnp.maximum(ok.sum(0), 1)
    n_g = jnp.maximum((ok.sum(0) > 0).sum(), 1)
    L = per_g.sum() / n_g
    if LAM_V > 0:  # seasonal volume: 91-day blocks of the scored window (728 = 8 x 91)
        okf = ok.astype(q.dtype)
        n_b = okf.reshape(SCORED // 91, 91, -1).sum(1)
        mq = (jnp.maximum(q, 0) * okf).reshape(SCORED // 91, 91, -1).sum(1) / jnp.maximum(n_b, 1)
        mo = (o * okf).reshape(SCORED // 91, 91, -1).sum(1) / jnp.maximum(n_b, 1)
        eps = 0.1 * d["obs_sd"][None, :]
        vb = n_b >= 46
        L = L + LAM_V * jnp.where(vb, (jnp.log(mq + eps) - jnp.log(mo + eps)) ** 2, 0.0).sum() / jnp.maximum(vb.sum(), 1)
    return L


def penalties(params, d, pen, bname):
    L = 0.0
    if pen is not None:
        g = jax.nn.softplus(params["gain"][bname]) * d["vmean"]
        L = L + LAM_L * pen + LAM_G * jnp.mean(g / d["vmean"])
    if DAM_EMB:
        L = L + LAM_E * jnp.mean(params["dam_emb"][bname] ** 2)
    return L


def storage_err(S, d, t0):
    """Per dam NSE-style SWOT storage-anomaly error on the scored days of the window, and which dams count (>= 5
    observed days in the window)."""
    Sm = S[SPIN:]
    o = jax.lax.dynamic_slice_in_dim(d["s_obs"], t0 + SPIN, SCORED, 0)
    ok = jax.lax.dynamic_slice_in_dim(d["s_ok"], t0 + SPIN, SCORED, 0)
    n = ok.sum(0)
    nn = jnp.maximum(n, 1.0)
    ma = Sm - (Sm * ok).sum(0) / nn
    oa = o - (o * ok).sum(0) / nn
    return (((ma - oa) ** 2) * ok).sum(0) / nn / d["s_den"], n >= 5


def storage_corr(S, d, t0):
    """Per dam correlation of modelled and SWOT storage on the observed scored days of the window (>= 5 days)."""
    Sm = S[SPIN:] / d["res"]["cap"][None, :]
    o = jax.lax.dynamic_slice_in_dim(d["s_obs"], t0 + SPIN, SCORED, 0) / d["res"]["cap"][None, :]
    ok = jax.lax.dynamic_slice_in_dim(d["s_ok"], t0 + SPIN, SCORED, 0)
    n = ok.sum(0)
    nn = jnp.maximum(n, 1.0)
    ma = (Sm - (Sm * ok).sum(0) / nn) * ok
    oa = (o - (o * ok).sum(0) / nn) * ok
    rho = (ma * oa).sum(0) / jnp.sqrt(((ma ** 2).sum(0) + 1e-10 * nn) * ((oa ** 2).sum(0) + 1e-10 * nn))
    return rho, (n >= 5) & ((oa ** 2).sum(0) > 0)


def river_rho(V, d, t0):
    """Per reach correlation of sqrt(flow) and SWOT WSE on the reach's observed scored days of the window, and which
    reaches count (>= RSW_MIN_WIN days, WSE not constant)."""
    r = d["rsw"]
    R = r["tr"].shape[0]
    t = r["t"] - (t0 + SPIN)
    w = ((t >= 0) & (t < SCORED)).astype(V.dtype)
    Vs = V[SPIN:]
    q0 = 0.01 * jax.lax.stop_gradient(Vs.mean(0)) + 1e-6
    x = jnp.sqrt(jnp.maximum(Vs[jnp.clip(t, 0, SCORED - 1), r["n"]], 0.0) + q0[r["n"]])
    n = jax.ops.segment_sum(w, r["r"], num_segments=R)
    nn = jnp.maximum(n, 1.0)
    mx = jax.ops.segment_sum(w * x, r["r"], num_segments=R) / nn
    my = jax.ops.segment_sum(w * r["y"], r["r"], num_segments=R) / nn
    dx = (x / jnp.maximum(jax.lax.stop_gradient(mx), 1e-6)[r["r"]] - 1.0) * w  # relative, so the floor below is scale-free
    dy = (r["y"] - my[r["r"]]) * w
    vx = jax.ops.segment_sum(dx * dx, r["r"], num_segments=R)
    vy = jax.ops.segment_sum(dy * dy, r["r"], num_segments=R)
    rho = jax.ops.segment_sum(dx * dy, r["r"], num_segments=R) / jnp.sqrt((vx + 1e-8 * nn) * (vy + 1e-8 * nn))
    return rho, (n >= RSW_MIN_WIN) & (vy > 1e-6 * nn)


def river_scale(theta):
    return RSW_SMIN + (RSW_SMAX - RSW_SMIN) * jax.nn.sigmoid(theta)


def loss_swot(params, d, n_seg, n_lev, t0, bname):
    V, pen, S = forward(params, d, n_seg, n_lev, t0, bname)
    L = penalties(params, d, pen, bname)
    if SWOT and "s_obs" in d:
        if SWOT_FORM == "corr":
            rho, valid = storage_corr(S, d, t0)
            Ls = jnp.where(valid, 1.0 - rho, 0.0).sum() / jnp.maximum(valid.sum(), 1)
        else:
            per, valid = storage_err(S, d, t0)
            # log1p: robust per dam (the c2wva prior misses some run-of-river pools by 30-400x their SWOT variance; smoke 24868167)
            Ls = jnp.where(valid, jnp.log1p(per), 0.0).sum() / jnp.maximum(valid.sum(), 1)
        L = L + LAM_S * Ls
    if RSW and "rsw" in d:
        rho, valid = river_rho(V, d, t0)
        sc = river_scale(params["rsw"][bname][: rho.shape[0]])
        use = valid & d["rsw"]["tr"]
        L = L + LAM_R * jnp.where(use, (1.0 - rho) / sc + jnp.log(sc), 0.0).sum() / jnp.maximum(use.sum(), 1)
    return L


optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(float(opt.get("lr", 3e-3))))
p_tr, p_static = eqx.partition(params, eqx.is_inexact_array)
state = optim.init(p_tr)
steps, evals, steps_sw, evals_sw = {}, {}, {}, {}
for b in B:
    n_seg, n_lev = b.st["n_seg"], b.st["n_lev"]
    gm = np.zeros(len(b.gauges), bool)
    gm[: len(b.g_train)] = True  # gauges = train, val, test in that order
    gmask = jnp.asarray(gm & np.isfinite(b.obs_sd))

    def make(n_seg=n_seg, n_lev=n_lev, gmask=gmask, bname=b.basin):
        @eqx.filter_jit
        def step(p_tr, state, d, t0, S0=None, E0=None):
            l, gr = jax.value_and_grad(lambda p: loss_fn(eqx.combine(p, p_static), d, n_seg, n_lev, t0, gmask, bname, S0, E0))(p_tr)
            up, state = optim.update(gr, state, p_tr)
            return optax.apply_updates(p_tr, up), state, l

        @eqx.filter_jit
        def pred(p_tr, d, t0, S0=None, E0=None):
            V, _, S, F = forward(eqx.combine(p_tr, p_static), d, n_seg, n_lev, t0, bname, S0, E0, flux=True)
            return V[SPIN:][:, d["gauge"]] / d["gup"][None, :], S, (F[..., 3:] if (LONG and F is not None) else None)

        @eqx.filter_jit
        def step_sw(p_tr, state, d, t0):
            l, gr = jax.value_and_grad(lambda p: loss_swot(eqx.combine(p, p_static), d, n_seg, n_lev, t0, bname))(p_tr)
            up, state = optim.update(gr, state, p_tr)
            return optax.apply_updates(p_tr, up), state, l

        @eqx.filter_jit
        def pred_sw(p_tr, d, t0):
            V, _, S = forward(eqx.combine(p_tr, p_static), d, n_seg, n_lev, t0, bname)
            out = {}
            if SWOT and "s_obs" in d:
                out["s_err"], out["s_valid"] = storage_err(S, d, t0)
                out["s_rho"], out["s_rvalid"] = storage_corr(S, d, t0)
            if RSW and "rsw" in d:
                out["r_rho"], out["r_valid"] = river_rho(V, d, t0)
            return out

        return step, pred, step_sw, pred_sw

    steps[b.basin], evals[b.basin], steps_sw[b.basin], evals_sw[b.basin] = make()

w_basin = np.array([len(b.g_train) for b in B], float)
w_basin /= w_basin.sum()
w_sw = np.zeros(len(B))  # SWOT-era windows: basins in proportion to their SWOT dams and / or river reaches
for cnt in ([b.n_sdam for b in B] if SWOT else None, [b.n_rsw_train for b in B] if RSW else None):
    if cnt is not None and sum(cnt) > 0:
        w_sw += np.array(cnt, float) / sum(cnt)
w_sw = w_sw / w_sw.sum() if w_sw.sum() > 0 else w_sw


bank = {}  # --carry=1: basin -> (S (days, D), EMA (days, D, 3) or None) at the start of each day


def evaluate(p_tr):
    out = {}
    for b in B:
        starts = list(range(0, train_end[b.basin] - TW + 1, SCORED))
        starts = [7 * (s0 // 7) for s0 in starts]
        chain = CARRY and len(b.dam_node) > 0
        if chain:
            assert all(np.diff(starts) == SCORED), "windows not back to back"
            nd = train_end[b.basin] + 1
            bS = np.full((nd, len(b.dam_node)), np.nan, np.float32)
            bE = np.full((nd, len(b.dam_node), 3), np.nan, np.float32) if LONG else None
            S0, E0 = 0.5 * DD[b.basin]["res"]["cap"], None
        qs, os_ = [], []
        for k, t0 in enumerate(starts):
            if chain:
                q, S, F = evals[b.basin](p_tr, DD[b.basin], jnp.int32(t0), S0, E0)
                last = k == len(starts) - 1
                n_keep = TW if last else SCORED  # S[j] = storage at the start of day t0 + j + 1
                bS[t0 + 1: t0 + 1 + n_keep] = np.asarray(S[:n_keep])
                if k == 0:
                    bS[0] = np.asarray(S0)
                S0 = S[SCORED - 1]
                if LONG:
                    bE[t0 + 1: t0 + 1 + n_keep] = np.asarray(F[:n_keep])
                    if k == 0:
                        bE[0] = np.asarray(DD[b.basin]["res"]["iref"])[:, None]
                    E0 = F[SCORED - 1]
            else:
                q, _, _ = evals[b.basin](p_tr, DD[b.basin], jnp.int32(t0))
            qs.append(np.asarray(q))
            os_.append(b.obs[t0 + SPIN: t0 + SPIN + SCORED])
        if chain:
            assert np.isfinite(bS[: train_end[b.basin] - TW + 1]).all(), "bank has gaps"
            bank[b.basin] = (bS, bE)
            fill = bS[: train_end[b.basin]] / b.dam_cap[None, :]
            out.setdefault("bank", {})[b.basin] = {"fill_med": round(float(np.median(np.nanmean(fill, 0))), 3),
                                                   "ann_range_med": round(float(np.median(
                                                       pd.DataFrame(fill).groupby(np.arange(len(fill)) // 365).mean()
                                                       .pipe(lambda x: x.max() - x.min()))), 3)}
        q, o = np.concatenate(qs), np.concatenate(os_)
        res = {}
        for name, sl in [("train", slice(0, len(b.g_train))),
                         ("val", slice(len(b.g_train), len(b.g_train) + len(b.g_val)))]:
            qq, oo = q[:, sl], o[:, sl]
            ok = np.isfinite(oo)
            nse, alb = [], []
            for j in range(qq.shape[1]):
                if ok[:, j].sum() >= 365:
                    x, y = qq[ok[:, j], j], oo[ok[:, j], j]
                    nse.append(1 - ((x - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
                    alb.append(abs(np.log(max(x.mean(), 1e-9) / max(y.mean(), 1e-9))))
            res[name] = float(np.median(nse)) if nse else float("nan")
            res[f"{name}_alb"] = float(np.median(alb)) if alb else float("nan")  # median |log beta|
        out[b.basin] = res
        if (SWOT and b.n_sdam) or (RSW and b.n_rsw):  # SWOT fit on the last window of the record (scored 2024-01..2025-12)
            o_sw = {k: np.asarray(v) for k, v in evals_sw[b.basin](p_tr, DD[b.basin], jnp.int32(7 * ((len(b.dates) - TW) // 7))).items()}
            if "s_err" in o_sw:  # in-sample
                per = o_sw["s_err"][o_sw["s_valid"]]
                res["swot_nse"] = float(np.median(1 - per)) if len(per) else float("nan")
                rr = o_sw["s_rho"][o_sw["s_rvalid"]]
                res["swot_rho"] = float(np.median(rr)) if len(rr) else float("nan")
            if "r_rho" in o_sw:  # river reaches: in the loss (train) and held out
                tr_ = np.asarray(DD[b.basin]["rsw"]["tr"])
                for nm_, m_ in (("rsw_rho_train", o_sw["r_valid"] & tr_), ("rsw_rho_hold", o_sw["r_valid"] & ~tr_)):
                    res[nm_] = float(np.median(o_sw["r_rho"][m_])) if m_.any() else float("nan")
                res["rsw_s_med"] = float(np.median(np.asarray(river_scale(p_tr["rsw"][b.basin]))[: len(tr_)][tr_]))
    out["val_mean_over_basins"] = float(np.nanmean([out[b.basin]["val"] for b in B]))
    if SWOT:
        out["swot_nse_mean_over_basins"] = float(np.nanmean([out[b.basin].get("swot_nse", np.nan) for b in B]))
        out["swot_rho_mean_over_basins"] = float(np.nanmean([out[b.basin].get("swot_rho", np.nan) for b in B]))
    if RSW:
        for nm_ in ("rsw_rho_train", "rsw_rho_hold"):
            out[f"{nm_}_mean_over_basins"] = float(np.nanmean([out[b.basin].get(nm_, np.nan) for b in B]))
    out["val_alb_mean_over_basins"] = float(np.nanmean([out[b.basin]["val_alb"] for b in B]))
    return out


if "init" in opt:
    p_tr = {**p_tr, **eqx.tree_deserialise_leaves(opt["init"], {k: v for k, v in p_tr.items() if k != "rsw"})}

if opt.get("diag") == "1":
    # gradient norms by component, and what the release policy does to storage, on 3 training windows per basin
    def gnorm(t):
        return float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(t)))) if t is not None else None
    for b in B:
        n_seg, n_lev = b.st["n_seg"], b.st["n_lev"]
        gm = np.zeros(len(b.gauges), bool)
        gm[: len(b.g_train)] = True
        gmask = jnp.asarray(gm & np.isfinite(b.obs_sd))
        lf = eqx.filter_jit(lambda p, t0: jax.value_and_grad(
            lambda q: loss_fn(eqx.combine(q, p_static), DD[b.basin], n_seg, n_lev, t0, gmask, b.basin))(p))
        for t0 in [7 * 52 * 10, 7 * 52 * 25]:
            l, g = lf(p_tr, jnp.int32(t0))
            m = g["model"]
            d = {"basin": b.basin, "t0": t0, "loss": float(l), "g_route": gnorm(m.route_net), "g_loss": gnorm(m.loss_net),
                 "g_policy": gnorm(m.policy), "g_gain": gnorm(g["gain"][b.basin]) if g["gain"] is not None else None}
            if USE_RES and len(b.dam_node):
                _, S, _ = evals[b.basin](p_tr, DD[b.basin], jnp.int32(t0))
                S = np.asarray(S)[SPIN:] / b.dam_cap[None, :]
                rng_ = S.max(0) - S.min(0)
                d.update({"S/cap mean pct10/50/90": np.percentile(S.mean(0), [10, 50, 90]).round(3).tolist(),
                          "S range/cap pct10/50/90": np.percentile(rng_, [10, 50, 90]).round(4).tolist(),
                          "dams with range < 1% cap": int((rng_ < 0.01).sum()), "dams": int(len(rng_))})
            log(d)
    sys.exit(0)



def ckpt(p):
    """Checkpoint tree without the per-reach noise logits (same leaves as a run without --rsw, so ladder_diag,
    da_lakesp and swot_era_flows read it unchanged)."""
    return {k: v for k, v in p.items() if k != "rsw"}


t0_run, it, best = time.time(), 0, -np.inf
if CARRY:  # the first bank, from the initial (or --init) parameters
    log({"step": 0, "event": "bank", **evaluate(p_tr)})
hist_sw = []
budget = 60 * float(opt.get("minutes", 60))
eval_every = int(opt.get("eval_every", 200))
hist = []
while True:
    if (SWOT or RSW) and rng.random() < P_SW:  # a SWOT-era window: SWOT terms only (no gauge after 2022)
        b = B[rng.choice(len(B), p=w_sw)]
        lo, hi = sw_range[b.basin]
        t0 = 7 * rng.integers(-(-lo // 7), hi // 7 + 1)
        p_tr, state, l = steps_sw[b.basin](p_tr, state, DD[b.basin], jnp.int32(t0))
        hist_sw.append(float(l))
    else:
        k = rng.choice(len(B), p=w_basin)
        b = B[k]
        t0 = 7 * rng.integers(0, (train_end[b.basin] - TW) // 7 + 1)
        # t0 as a device scalar: a Python int is static under filter_jit and recompiled the step for every new window
        # start (2026-10-21; the s1s_f0 ladder grew to 96 GB host RAM and was OOM-killed after 5-6 h)
        if CARRY and b.basin in bank:
            bS, bE = bank[b.basin]
            p_tr, state, l = steps[b.basin](p_tr, state, DD[b.basin], jnp.int32(t0), jnp.asarray(bS[t0]),
                                            None if bE is None else jnp.asarray(bE[t0]))
        else:
            p_tr, state, l = steps[b.basin](p_tr, state, DD[b.basin], jnp.int32(t0))
        hist.append(float(l))
    it += 1
    if it % 50 == 0:
        log({"step": it, "minutes": round((time.time() - t0_run) / 60, 1), "loss_last50": round(float(np.mean(hist[-50:])), 4),
             **({"swot_loss_last20": round(float(np.mean(hist_sw[-20:])), 4)} if hist_sw else {})})
    done = time.time() - t0_run > budget
    if it % eval_every == 0 or done:
        ev = evaluate(p_tr)
        log({"step": it, "minutes": round((time.time() - t0_run) / 60, 1), "train_loss": round(float(np.mean(hist)), 4),
             **({"swot_loss": round(float(np.mean(hist_sw)), 4)} if hist_sw else {}), **{k2: v for k2, v in ev.items()}})
        hist, hist_sw = [], []
        eqx.tree_serialise_leaves(OUT / "params_last.eqx", ckpt(p_tr))
        if ev["val_mean_over_basins"] > best:
            best = ev["val_mean_over_basins"]
            eqx.tree_serialise_leaves(OUT / "params_best.eqx", ckpt(p_tr))
    if done:
        break
log({"event": "done", "steps": it, "best_val_mean_over_basins": best})
