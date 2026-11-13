"""Model flow in the SWOT period at the SWOT-reach nodes and the gauges (2026-11-12; inference only, no training;
input of swot_rank.py, the SWOT river rank diagnostic).

Each run (--runs=name:rundir:ckpt,...) is run once per basin over one window, as da_lakesp.py's prior: 364-day spin-up
from ~2022-07 (dams half full at its start), then every day from 2023-07-01 to the end of the record. Forward as
ladder_diag.make_fn (same options read from each run's config: loss form, demand, reservoirs, evaporation, area,
dam embedding, downstream demand, long memory), in two modes for loss runs:
  full   = as trained;  noloss = withdrawals / losses off, gains and reservoirs kept (same routing: the paired control).
Nothing of the 2023+ window is used for any choice here. Carried-storage runs (--carry=1, arm P) are not supported
(their dams would need the chained 1980-2022 state).
Writes --out/flows_<basin>.npz: dates, nodes (dataset node names: nodes of matched SWORD reaches, then all gauges),
is_gauge, sets, uparea, and per run and mode "<run>__<mode>" (T, n) m3/s; obs (T, n_gauges) m3/s at the gauges (NaN gaps).
--local=1 (local toggle, swot_local.py): per loss run and mode also "<run>__<mode>__rm" / "__ro" (T, n_reach_nodes): the
flow of the nearest upstream gauges (no model dam between) routed to each SWOT node with the run's kernels, from the
model (rm) and from the observations (ro); full mode with the path pass-through, noloss without. tf_bad, tf_n_up.
Usage: python swot_era_flows.py --out=DIR --runs=name:rundir:ckpt[,...] [--fold=0] [--order=basin,...] [--skip_done=1] [--local=1]
"""
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, "/project/swot-ml/swot-ml/overhaul/stage2")
from s2_core import (KM2_MM_TO_M3S, LOSS_STATICS, ROUTE_STATICS, VAR_FLOOR, Basin, S2Model, _conv_series, _std,  # noqa: E402
                     downstream_demand, l2_demand, l2_pass, r2_route_res)

opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
OUT = Path(opt["out"])
OUT.mkdir(parents=True, exist_ok=True)
OV = "/project/swot-ml/data/overhaul"
RUNS_DIR = Path("/project/swot-ml/swot-ml/overhaul/stage2/runs")
BASINS = "USGS-07289000,USGS-09521100,USGS-14246900,USGS-02359170,USGS-01578310".split(",")  # train_s2 order (scales)
ORDER = opt.get("order", "USGS-01578310,USGS-02359170,USGS-09521100,USGS-14246900,USGS-07289000").split(",")
RUNS = [r.split(":") for r in opt["runs"].split(",")]
FOLD = int(opt.get("fold", 0))
SPIN = 364
LOCAL = opt.get("local", "0") == "1"  # also save the routed upstream-gauge flows (swot_local.py)
RUNOFF = f"{OV}/stage2/runoff_s1s_ens"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


B = [Basin(b, RUNOFF, fold=FOLD, folds="lists_topo_qc", route_init=f"{OV}/stage2/route_init_s1s.npz", route_init_mode="hybrid")
     for b in BASINS]
ref_r = _std(pd.concat([b.A[ROUTE_STATICS] for b in B]))[1]
ref_l = _std(pd.concat([b.A[LOSS_STATICS] for b in B]))[1]
drv_all = np.concatenate([b.drv.reshape(-1, 2) for b in B[:1]])
drv_mu, drv_sd = drv_all.mean(0), drv_all.std(0) + 1e-9
train_end = {b.basin: int(np.searchsorted(b.dates, np.datetime64("2022-12-31"), side="right")) for b in B}
RM = pd.read_parquet(f"{OV}/swot_rivers/reach_match.parquet")
RM = RM[RM.why == "ok"]
log(f"loaded {len(B)} basins; matched SWORD reaches {len(RM)}")


def device_data(b, use_res, res_evap, res_area):
    st = {k: jnp.asarray(v) for k, v in b.st.items() if isinstance(v, np.ndarray)}
    d = {"st": st, "xs": jnp.asarray(_std(b.A[ROUTE_STATICS], ref_r)[0]), "xl": jnp.asarray(_std(b.A[LOSS_STATICS], ref_l)[0]),
         "tau": jnp.asarray(b.tau), "route_base": {k: jnp.asarray(v) for k, v in b.route_base.items()}, "vol": jnp.asarray(b.vol),
         "drv": jnp.asarray((b.drv - drv_mu) / drv_sd), "vmean": jnp.asarray(b.vmean),
         "doy": jnp.asarray(2 * np.pi * b.dates.dayofyear.values / 365.25, jnp.float32), "a_use": jnp.asarray(b.a_use),
         "wscale": jnp.asarray((b.area * b.drv[: train_end[b.basin] // 7, :, 0].mean(0) * KM2_MM_TO_M3S).astype(np.float32))}
    if use_res and len(b.dam_node):
        d["res"] = {"seg": jnp.asarray(b.dam_seg), "node": jnp.asarray(b.dam_node), "cap": jnp.asarray(b.dam_cap),
                    "iref": jnp.asarray(b.dam_iref), "feat": jnp.asarray(b.dam_feat), "lv_idx": jnp.asarray(b.dam_lv_idx)}
        if res_evap:
            d["res"]["area"] = jnp.asarray(b.dam_area_max if res_area == "max" else b.dam_area)
            d["res_pet"] = jnp.asarray(np.maximum(b.dam_pet, 0.0))
        d["dsd_pairs"] = {k: jnp.asarray(v) for k, v in b.dsd_pairs.items()}
    return d


def make_fn(b, t0, TW, use_loss, mode, l2, wcap, dsd, lm, demand):
    n_seg, n_lev, bname = b.st["n_seg"], b.st["n_lev"], b.basin

    @eqx.filter_jit
    def fn(params, d):
        m = params["model"]
        vol = d["vol"][t0:t0 + TW]
        s = None
        if use_loss:
            drv = d["drv"][t0 // 7:t0 // 7 + TW // 7]
            doy_w = d["doy"][t0:t0 + TW][3::7]
            sc_w = jnp.stack([jnp.sin(doy_w), jnp.cos(doy_w)], -1)
            if l2:
                d_free, d_irr = l2_demand(m, d["xl"], drv, sc_w, d["wscale"], demand, d["a_use"],
                                          drv[..., 0] * float(drv_sd[0]) + float(drv_mu[0]))
                dem_w = sum(x for x in (d_free, d_irr) if x is not None)
                s = jnp.repeat(dem_w, 7, axis=0)
            else:
                s = jnp.repeat(m.loss_field(d["xl"], drv, sc_w), 7, axis=0)
            vol = vol + (jax.nn.softplus(params["gain"][bname]) * d["vmean"])[None, :]
        prm = m.route_params(d["xs"], d["tau"], d["route_base"])
        res = None
        if "res" in d:
            doy = d["doy"][t0:t0 + TW]
            res = dict(d["res"], sc=jnp.stack([jnp.sin(doy), jnp.cos(doy)], -1))
            if "dam_emb" in params:
                res["feat"] = jnp.concatenate([res["feat"], params["dam_emb"][bname][: res["feat"].shape[0]]], -1)
            if "res_pet" in d:
                res["pet"] = d["res_pet"][t0:t0 + TW]
            if dsd:
                res["dsd"] = downstream_demand(dem_w, d["dsd_pairs"], d["res"]["iref"], TW)
            res["long_mem"] = lm
        if use_loss and l2 and mode == "full":
            V0, _ = r2_route_res(vol, prm, d["st"], n_seg, n_lev, res=res, policy=m.policy)
            s = l2_pass(s, V0, wcap)
        V, _ = r2_route_res(vol, prm, d["st"], n_seg, n_lev, res=res, policy=m.policy, s=s if mode == "full" else None)
        return V, s, prm

    return fn


def local_pairs(b, rn_idx, ok_w):
    """Pairs (upstream gauge u -> SWOT node n) for the local toggle (--local=1): every gauge with >= 365 observed days
    in the scored window is walked downstream until a model dam (not entered) or the next such gauge (included);
    each SWOT-reach node met on the way pairs with u, with the nodes of the path (u, n] for the kernel and the
    pass-through (as s2_core.tf_pairs / ladder_diag.diag_pairs, with interior nodes as targets)."""
    good = ok_w[SPIN:].sum(0) >= 365
    gpos = {int(n): j for j, n in enumerate(b.gauges) if good[j]}
    isdam = set(int(i) for i in b.dam_node)
    tpos = {int(n): k for k, n in enumerate(rn_idx)}
    pu, pt, pn, pp = [], [], [], []
    for u_node, ju in gpos.items():
        x, path = int(b.down[u_node]), []
        while x >= 0 and x not in isdam:
            path.append(x)
            if x in tpos:
                k = len(pu)
                pu.append(ju)
                pt.append(tpos[x])
                pn += path
                pp += [k] * len(path)
            if x in gpos:
                break
            x = int(b.down[x])
    return (np.asarray(pu, np.int32), np.asarray(pt, np.int32), np.asarray(pn, np.int32), np.asarray(pp, np.int32))


def local_tf(V, s, prm, pairs, obs_w, ok_w, gauges, n_t, max_path=120_000):
    """Routed upstream-gauge flow at the SWOT nodes: model (rm) and observed (ro), with the path pass-through when s is
    given; bad = a day whose kernel holds an unobserved upstream day. (TW, n_t) each."""
    pu, pt, pn, pp = pairs
    TWl = V.shape[0]
    rm, ro, bad = (jnp.zeros((TWl, n_t), V.dtype) for _ in range(3))
    starts = np.r_[np.searchsorted(pp, np.arange(len(pu))), len(pn)]  # path slice of each pair (pp is sorted)
    k0 = 0
    while k0 < len(pu):
        k1 = k0 + 1
        while k1 < len(pu) and starts[k1 + 1] - starts[k0] <= max_path and k1 - k0 < 1500:
            k1 += 1
        P = k1 - k0
        pn_c, pp_c = jnp.asarray(pn[starts[k0]:starts[k1]]), jnp.asarray(pp[starts[k0]:starts[k1]] - k0)
        mu = jax.ops.segment_sum(prm["transit"][pn_c], pp_c, num_segments=P)
        sd = jnp.sqrt(jnp.maximum(jax.ops.segment_sum(prm["transit_var"][pn_c], pp_c, num_segments=P), VAR_FLOOR))
        u = jnp.asarray(pu[k0:k1])
        okf = ok_w[:, u]
        cm = _conv_series(V[:, gauges[u]], mu, sd, 64, "tri")
        co = _conv_series(jnp.where(okf, obs_w[:, u], 0.0), mu, sd, 64, "tri")
        cb = (_conv_series(okf.astype(V.dtype), mu, sd, 64, "tri") < 0.99).astype(V.dtype)
        if s is not None:
            pas = jnp.exp(jax.ops.segment_sum(s[:, pn_c].T, pp_c, num_segments=P).T)
            cm, co = cm * pas, co * pas
        t = jnp.asarray(pt[k0:k1])
        rm = rm + jax.ops.segment_sum(cm.T, t, num_segments=n_t).T
        ro = ro + jax.ops.segment_sum(co.T, t, num_segments=n_t).T
        bad = bad + jax.ops.segment_sum(cb.T, t, num_segments=n_t).T
        k0 = k1
    return rm, ro, bad > 0


for b in sorted([x for x in B if x.basin in ORDER], key=lambda x: ORDER.index(x.basin)):
    f_out = OUT / f"flows_{b.basin}.npz"
    if opt.get("skip_done", "0") == "1" and f_out.exists():
        continue
    t_b = time.time()
    idx = {s: i for i, s in enumerate(b.names)}
    rn = sorted(set(RM.node[RM.basin == b.basin]) & set(idx))
    sel = np.array([idx[n] for n in rn] + [int(g) for g in b.gauges], np.int32)
    t_sw0 = int(np.searchsorted(b.dates, np.datetime64("2023-07-01")))
    t0 = 7 * ((t_sw0 - SPIN) // 7)
    TW = 7 * ((len(b.dates) - t0) // 7)
    days = np.arange(t0 + SPIN, t0 + TW)
    save = {"dates": b.dates[days].values.astype("datetime64[D]"), "nodes": b.names[sel],
            "is_gauge": np.r_[np.zeros(len(rn), bool), np.ones(len(b.gauges), bool)], "uparea": b.up_area[sel],
            "sets": np.array(["reach"] * len(rn) + ["train"] * len(b.g_train) + ["val"] * len(b.g_val) + ["test"] * len(b.g_test)),
            "obs": (b.obs[days] * b.up_area[b.gauges][None, :]).astype(np.float32)}
    sel_j = jnp.asarray(sel)
    if LOCAL:
        ow = (b.obs[t0:t0 + TW] * b.up_area[b.gauges][None, :]).astype(np.float32)
        okw = np.isfinite(ow)
        pairs = local_pairs(b, sel[:len(rn)], okw)
        obs_w, ok_w, gj = jnp.asarray(np.nan_to_num(ow)), jnp.asarray(okw), jnp.asarray(b.gauges)
        save["tf_n_up"] = np.bincount(pairs[1], minlength=len(rn)).astype(np.int32)
        log(f"{b.basin}: local pairs {len(pairs[0])}, path nodes {len(pairs[2])}, SWOT nodes with an upstream gauge "
            f"{int((save['tf_n_up'] > 0).sum())} of {len(rn)}")
    dd = {}
    for name, rdir, ck in RUNS:
        cfg = json.load(open(RUNS_DIR / rdir / "config.json"))
        assert int(cfg.get("fold", 0)) == FOLD, f"{rdir}: fold {cfg.get('fold')}"
        assert Path(cfg.get("runoff", RUNOFF)).resolve() == Path(RUNOFF).resolve(), f"{rdir}: runoff {cfg.get('runoff')}"
        assert cfg.get("carry", "0") != "1", f"{rdir}: carried-storage runs are not supported here"
        assert float(cfg.get("dam_min", 100)) == 100
        use_loss, use_res = cfg.get("loss", "0") == "1", cfg.get("res", "0") == "1"
        l2, wcap = cfg.get("loss_form", "l1") == "l2", float(cfg.get("wcap", 0.9))
        res_evap, res_area = use_res and cfg.get("res_evap", "0") == "1", cfg.get("res_area", "poly")
        key = (use_res, res_evap, res_area)
        if key not in dd:
            dd[key] = device_data(b, use_res, res_evap, res_area)
        d = dd[key]
        n_emb = int(cfg.get("dam_emb", 0)) if use_res else 0
        dsd = use_res and use_loss and l2 and cfg.get("dsd", "0") == "1"
        lm = use_res and cfg.get("long_mem", "0") == "1"
        demand = cfg.get("demand", "free")
        model = S2Model(len(ROUTE_STATICS), len(LOSS_STATICS), b.dam_feat.shape[1] + n_emb + 2 * dsd + 2 * lm, use_loss, use_res,
                        jax.random.PRNGKey(0), use_irr=demand != "free")
        params = {"model": model, "gain": {bb.basin: jnp.full(bb.N, -6.0) for bb in B} if use_loss else None}
        if n_emb:
            params["dam_emb"] = {bb.basin: jnp.zeros((max(len(bb.dam_node), 1), n_emb)) for bb in B}
        p_tr, p_static = eqx.partition(params, eqx.is_inexact_array)
        params = eqx.combine(eqx.tree_deserialise_leaves(RUNS_DIR / rdir / f"{ck}.eqx", p_tr), p_static)
        for mode in (["full", "noloss"] if use_loss else ["full"]):
            fn = make_fn(b, t0, TW, use_loss, mode, l2, wcap, dsd, lm, demand)
            Vj, sj, prm = fn(params, d)
            v = np.asarray(Vj[SPIN:][:, sel_j])
            if LOCAL and use_loss and len(pairs[0]):  # routed upstream-gauge flow at the SWOT nodes (full: with pass-through)
                rm, ro, bad = local_tf(Vj, sj if mode == "full" else None, prm, pairs, obs_w, ok_w, gj, len(rn))
                save[f"{name}__{mode}__rm"] = np.asarray(rm[SPIN:]).astype(np.float32)
                save[f"{name}__{mode}__ro"] = np.asarray(ro[SPIN:]).astype(np.float32)
                save["tf_bad"] = np.asarray(bad[SPIN:]) | save.get("tf_bad", False)
                del rm, ro, bad
            del Vj, sj
            assert np.isfinite(v).all(), f"{name} {mode}: non-finite flow"
            save[f"{name}__{mode}"] = v.astype(np.float32)
            log(f"{b.basin} {name} {mode}: {v.shape}, largest mean flow {v.mean(0).max():.0f} m3/s")
            del fn
        jax.clear_caches()
    np.savez_compressed(f_out, **save)
    log(f"{b.basin}: {len(rn)} reach nodes + {len(b.gauges)} gauges, {len(days)} days "
        f"({save['dates'][0]} .. {save['dates'][-1]}), {time.time() - t_b:.0f} s")
