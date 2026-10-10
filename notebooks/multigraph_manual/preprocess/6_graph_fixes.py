"""
Graph fixes v2 (2026-10-10): gauge snapping, Murray side-channel gauges and duplicated GDW dams.

Every output is new; the v1 files stay as they are (runs in flight read them).

1. Gauges (gauge_fixes_v2.csv, one row per gauge, with the reason):
   - resnap: the gauge was matched to a small tributary node next to its river. Its observations move to
     the main-stem node with the right upstream area (discharge copied; unit_discharge = discharge /
     the new node's upstream area). The old node keeps its forcings but loses its observations.
   - remove: no node of the right size exists near the gauge (the graph's merged nodes skip it, or the
     only candidate is another gauge), or the gauge is on a Murray-Darling side channel (effluent creek,
     anabranch, distributary, off-river storage outlet, canal) that receives water through a bifurcation
     the tree graph doesn't have. Gauges that lose water to such channels stay: routing losses can
     represent that.
   Basin lists: <name>_v2.csv for every list that holds an affected gauge (resnapped gauges appear under
   their new node id). Zarr: <zbs>_v2/ symlinks every basin store; a basin with resnaps gets a new store
   that symlinks every variable except discharge and unit_discharge, which are copied and edited.

2. Dams: 2_hydroatlas_attributes.ipynb joined GDW reservoir polygons to 2.5 km-buffered subbasins and
   deduplicated per subbasin, so a reservoir that overlaps several subbasins gave each of them the full
   capacity (Mississippi: 1,414,069 MCM summed vs 253,144 MCM over distinct dams). static_v2.parquet keeps
   each dam (same GDW properties within a basin) only at its most downstream node (largest upstream area),
   then 5_graph_attributes.py rebuilds static_graph_v2.parquet. Dams lost by that join (one reservoir per
   subbasin: John Martin, Pueblo, Hebgen, ...) need the subbasin polygons and GDW barrier points to
   restore; not done here.

Usage: python 6_graph_fixes.py [lists] [zarr] [dams]   (all three by default; zarr copies ~1 GB, run via sbatch)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path("/project/swot-ml/swot-ml/data/multigraph_manual")
ZBS = Path("/project/swot-ml/data/zbs")
ZBS_V2 = Path("/project/swot-ml/data/zbs_v2")
FIXES = HERE / "gauge_fixes_v2.csv"
GDW_PROPS = ["DAM_TYPE", "YEAR_DAM", "AREA_POLY", "CAP_MCM", "DIS_AVG_LS", "DOR_PC"]


def load_fixes() -> pd.DataFrame:
    A = pd.read_csv(FIXES, dtype=str).fillna("")
    assert A.site_id.is_unique, "a gauge appears twice in gauge_fixes_v2.csv"
    assert set(A.action) <= {"resnap", "remove"}
    return A


def fix_lists(A: pd.DataFrame):
    drop = set(A.loc[A.action == "remove", "site_id"])
    move = dict(A.loc[A.action == "resnap", ["site_id", "new_node"]].values)
    for f in sorted((DATA / "basin_lists").glob("*.csv")):
        if f.stem.endswith("_v2"):
            continue
        df = pd.read_csv(f, dtype=str)
        hit = df.subbasin.isin(drop) | df.subbasin.isin(move)
        if not hit.any():
            continue
        out = df[~df.subbasin.isin(drop)].copy()
        clash = out.subbasin.isin(set(move.values()))
        assert not clash.any(), f"{f.name}: resnap target already listed: {out.subbasin[clash].tolist()}"
        out["subbasin"] = out.subbasin.replace(move)
        out.to_csv(f.with_name(f"{f.stem}_v2.csv"), index=False)
        print(f"{f.name}: {len(df)} -> {len(out)} rows ({df.subbasin.isin(drop).sum()} removed, "
              f"{df.subbasin.isin(move).sum()} resnapped)")


def fix_zarr(A: pd.DataFrame):
    import xarray as xr
    import zarr

    up = pd.read_parquet(DATA / "attributes/static_graph.parquet", columns=["uparea_km2"]).uparea_km2
    R = A[A.action == "resnap"]
    ZBS_V2.mkdir(exist_ok=True)
    for entry in sorted(ZBS.iterdir()):
        if entry.name.startswith("_cache") or entry.name in set(R.basin):
            continue
        link = ZBS_V2 / entry.name
        if not link.exists():
            os.symlink(entry, link)
    for basin, rb in R.groupby("basin"):
        src, dst = ZBS / basin, ZBS_V2 / basin
        if dst.exists():
            raise FileExistsError(f"{dst} exists; delete it to rebuild")
        dst.mkdir()
        for child in src.iterdir():
            if child.name in ("discharge", "unit_discharge"):
                shutil.copytree(child, dst / child.name)
            elif child.is_file():
                shutil.copy2(child, dst / child.name)
            else:
                os.symlink(child, dst / child.name)
        with xr.open_zarr(src, consolidated=None) as z:
            sb = pd.Index(z.subbasin.values.astype(str))
        root = zarr.open_group(str(dst), mode="r+")
        Q, U = root["discharge"], root["unit_discharge"]
        for _, r in rb.iterrows():
            i, j = sb.get_loc(r.site_id), sb.get_loc(r.new_node)
            q, u = Q[:, i], U[:, i]
            assert np.all(np.isnan(Q[:, j])), f"{r.new_node} already has observations"
            ok = (q > 0) & (u > 0)
            area_old = float(np.median(q[ok] / u[ok]))  # the area the zarr divided by
            area_new = float(up.loc[(basin, r.new_node)])
            Q[:, j] = q
            U[:, j] = u * area_old / area_new
            Q[:, i] = np.nan
            U[:, i] = np.nan
            print(f"{basin} {r.site_id} -> {r.new_node}: area {area_old:.1f} -> {area_new:.1f} km2, "
                  f"{int(np.isfinite(q).sum())} days moved")
    print(f"zarr: {ZBS_V2}")


def fix_dams():
    st = pd.read_parquet(DATA / "attributes/static.parquet")
    up = pd.read_parquet(DATA / "attributes/static_graph.parquet", columns=["uparea_km2"]).uparea_km2
    dam = st.has_dam.fillna(False).astype(bool)
    d = st.loc[dam, GDW_PROPS].copy()
    d["uparea"] = up.reindex(d.index).values
    d = d.reset_index()
    key = ["basin"] + GDW_PROPS
    keep = d.sort_values("uparea").groupby(key, dropna=False).tail(1).set_index(["basin", "subbasin"]).index
    dup = st.index[dam].difference(keep)
    v2 = st.copy()
    v2.loc[dup, GDW_PROPS] = np.nan
    v2.loc[dup, "has_dam"] = False
    print(f"dams: {int(dam.sum())} dam nodes -> {len(keep)} ({len(dup)} duplicates cleared); "
          f"capacity summed {st.CAP_MCM[dam].sum():,.0f} -> {v2.CAP_MCM[v2.has_dam.fillna(False).astype(bool)].sum():,.0f} MCM")
    v2.to_parquet(DATA / "attributes/static_v2.parquet")
    script = HERE / "5_graph_attributes.py"
    subprocess.run([sys.executable, str(script), "--attributes", str(DATA / "attributes/static_v2.parquet"),
                    "--graph", str(DATA / "metadata/all_graphs.json"),
                    "--out", str(DATA / "attributes/static_graph_v2.parquet")], check=True)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["lists", "zarr", "dams"]
    A = load_fixes()
    if "lists" in steps:
        fix_lists(A)
    if "zarr" in steps:
        fix_zarr(A)
    if "dams" in steps:
        fix_dams()
