"""
Dam join v3 (2026-10-10): GDW v1.0 barrier points -> one graph node each, without subbasin polygons.

2_hydroatlas_attributes.ipynb overlaid GDW reservoir polygons with buffered subbasin polygons and kept one
reservoir per subbasin, so a reservoir was copied onto every subbasin it touched and other dams were lost
(John Martin, Pueblo, Hebgen, ...). 6_graph_fixes.py (v2) only removed the copies. Here every GDW barrier
with a reservoir (CAP_MCM > 0, the same set as the polygon layer) is placed on the graph from its own
location, using IDs the data already share:
  1. GDW gives each barrier its HydroBASINS level-12 unit (HYBAS_L12). Walk downstream along level-12
     NEXT_DOWN, mapping each unit to its level-9 parent (PFAF_ID // 1000). Every graph node carries the
     level-9 unit matched to its outlet (static.parquet HYBAS_ID). The candidates are the nodes in the
     first N9 level-9 units on the path that hold any node.
  2. Pick the candidate with the smallest upstream area >= LO x the dam's catchment (CATCH_SKM): the first
     node downstream of the dam that drains it.
  3. Level-9 units are coarse and a node's matched unit is approximate, so the pick is often a few nodes
     too far downstream. Refine: among the pick and its graph ancestors, take the node whose upstream area
     is closest to the dam's catchment within [LO, HI] x (keep the pick if none). Areas come from different
     DEMs (HydroSHEDS vs MERIT), so a tailwater gauge can be a little smaller than the catchment; when the
     choice is off it is usually one node upstream, which leaves every node below the dam correct.
Checked on the Mississippi against geometric snapping to the node reach lines (MS_network.geojson): for
dams > 100 MCM, 82% land on a node within 15% of the catchment area (geometric snapping: 65%).

Per node: has_dam, CAP_MCM (sum), AREA_POLY (sum), DIS_AVG_LS (max), and DAM_TYPE, YEAR_DAM, DOR_PC
from the largest dam; n_dams_gdw counts the dams. DAM_CATCH_SKM is the largest local dam catchment (any
dam, locks and run-of-river included); STORAGE_CATCH_SKM is the same over storage dams only (GDW DOR_PC
>= 5%: excludes run-of-river dams and navigation pools, keeps 95% of capacity).
Then 5_graph_attributes.py -> static_graph_v3.parquet. Given these it makes dam_local the share of the
node's area behind its largest local dam of any type (0..1, not a flag: 73% of placed dams are ponds on
side creeks draining ~1% of their node, and they alone flagged 53% of dam nodes; a main-stem lock reads
~1) and adds gu_reg_frac, the share of the node's area behind the largest single upstream storage dam.
Also writes attributes/gdw_dam_nodes_v3.csv (one row per placed dam, with the area ratio).

Inputs: /project/swot-ml/data/gdw (GDW_v1_0_shp from figshare 25988293) and /project/swot-ml/data/hydrobasins
(HydroBASINS standard v1c, levels 9 and 12, regions na ar au eu sa si).
Usage: python 7_dam_join.py
"""
import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyogrio

HERE = Path(__file__).resolve().parent
DATA = Path("/project/swot-ml/swot-ml/data/multigraph_manual")
GDW = Path("/project/swot-ml/data/gdw/GDW_v1_0_shp/GDW_barriers_v1_0.shp")
HB = Path("/project/swot-ml/data/hydrobasins")
REGIONS = ["na", "ar", "au", "eu", "sa", "si"]
N9, LO, HI, MAX_STEPS = 2, 0.85, 1.5, 2000
STORAGE_DOR_PC = 5.0  # a dam counts for DAM_CATCH_SKM if it stores >= 5% (about 2.5 weeks) of its mean inflow
GDW_PROPS = ["DAM_TYPE", "YEAR_DAM", "AREA_POLY", "CAP_MCM", "DIS_AVG_LS", "DOR_PC"]


def read_hybas(level, cols):
    return pd.concat([pyogrio.read_dataframe(HB / f"hybas_{r}_lev{level:02d}_v1c.dbf", read_geometry=False, columns=cols)
                      for r in REGIONS])


def place_dams(st: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    h12 = read_hybas(12, ["HYBAS_ID", "NEXT_DOWN", "PFAF_ID"]).set_index("HYBAS_ID")
    h9 = read_hybas(9, ["HYBAS_ID", "PFAF_ID"])
    pf9 = dict(zip(h9.PFAF_ID, h9.HYBAS_ID))
    to9 = {h: pf9.get(p // 1000) for h, p in h12.PFAF_ID.items()}
    nd12 = h12.NEXT_DOWN.to_dict()

    nodes = st.reset_index()[["basin", "subbasin", "HYBAS_ID", "uparea_km2"]]
    nodes["HYBAS_ID"] = nodes.HYBAS_ID.astype("int64")
    by9 = {h: list(zip(g.subbasin, g.uparea_km2)) for h, g in nodes.groupby("HYBAS_ID")}
    area = dict(zip(nodes.subbasin, nodes.uparea_km2))
    basin_of = dict(zip(nodes.subbasin, nodes.basin))

    gdw = pyogrio.read_dataframe(GDW, read_geometry=False,
                                 columns=["GDW_ID", "DAM_NAME", "RES_NAME"] + GDW_PROPS + ["CATCH_SKM", "HYBAS_L12"])
    gdw = gdw[(gdw.CAP_MCM > 0) & gdw.HYBAS_L12.isin(h12.index)]
    anc = {}
    rows = []
    for r in gdw.itertuples():
        h, units, cands, steps = int(r.HYBAS_L12), [], [], 0
        while h and len(units) < N9 and steps < MAX_STEPS:
            u = to9.get(h)
            if u is not None and u not in units and u in by9:
                units.append(u)
                cands += by9[u]
            h, steps = int(nd12.get(h, 0)), steps + 1
        if not cands:
            continue  # the dam drains to none of the graphs
        catch = float(r.CATCH_SKM) if r.CATCH_SKM > 0 else 0.0
        ok = [c for c in cands if c[1] >= LO * catch]
        if not ok:
            continue
        node = min(ok, key=lambda c: c[1])[0]
        if catch:
            if node not in anc:
                anc[node] = [node, *nx.ancestors(G, node)]
            win = [n for n in anc[node] if LO * catch <= area.get(n, 0) <= HI * catch]
            if win:
                node = min(win, key=lambda n: abs(np.log(area[n] / catch)))
        rows.append(dict(GDW_ID=r.GDW_ID, name=r.DAM_NAME or r.RES_NAME, basin=basin_of[node], subbasin=node,
                         node_uparea=area[node], catch_skm=catch, area_ratio=area[node] / catch if catch else np.nan,
                         **{c: getattr(r, c) for c in GDW_PROPS}))
    return pd.DataFrame(rows)


def node_attributes(st: pd.DataFrame, dams: pd.DataFrame) -> pd.DataFrame:
    v3 = st.copy()
    v3[GDW_PROPS] = np.nan
    v3["has_dam"] = False
    v3["n_dams_gdw"] = 0
    v3["DAM_CATCH_SKM"] = np.nan
    v3["STORAGE_CATCH_SKM"] = np.nan
    d = dams.copy()
    for c in GDW_PROPS[1:]:
        d[c] = d[c].where(d[c] >= 0)  # GDW no-data is -99
    d["catch_skm"] = d.catch_skm.where(d.catch_skm > 0)
    # Storage control counts only dams that store a real share of their inflow: run-of-river dams and
    # navigation pools have huge catchments but little storage (Bonneville, McNary: DOR_PC < 1%; Upper
    # Mississippi pools 2.5-4.7%). They still count everywhere else (has_dam, dam_local, capacity, n_dams).
    d["storage_catch_skm"] = d.catch_skm.where(d.DOR_PC >= STORAGE_DOR_PC)
    largest = d.sort_values("CAP_MCM").groupby(["basin", "subbasin"]).tail(1).set_index(["basin", "subbasin"])
    g = d.groupby(["basin", "subbasin"])
    agg = pd.DataFrame({"CAP_MCM": g.CAP_MCM.sum(), "AREA_POLY": g.AREA_POLY.sum(min_count=1),
                        "DIS_AVG_LS": g.DIS_AVG_LS.max(), "n_dams_gdw": g.size(), "DAM_CATCH_SKM": g.catch_skm.max(),
                        "STORAGE_CATCH_SKM": g.storage_catch_skm.max()})
    for c in ["DAM_TYPE", "YEAR_DAM", "DOR_PC"]:
        agg[c] = largest[c]
    idx = agg.index
    v3.loc[idx, list(agg.columns)] = agg.values
    v3.loc[idx, "has_dam"] = True
    v3["n_dams_gdw"] = v3.n_dams_gdw.astype(int)
    for c in ["YEAR_DAM", "AREA_POLY", "CAP_MCM", "DIS_AVG_LS", "DOR_PC", "DAM_CATCH_SKM", "STORAGE_CATCH_SKM"]:
        v3[c] = v3[c].astype(float)
    return v3


def main():
    st = pd.read_parquet(DATA / "attributes/static.parquet")
    assert st.index.get_level_values("subbasin").is_unique
    with open(DATA / "metadata/all_graphs.json") as f:
        G = nx.readwrite.json_graph.node_link_graph(json.load(f), edges="edges")
    dams = place_dams(st, G)
    dams.to_csv(DATA / "attributes/gdw_dam_nodes_v3.csv", index=False)
    r = dams.area_ratio
    big = dams.CAP_MCM > 100
    print(f"placed {len(dams)} GDW dams on {dams.groupby(['basin', 'subbasin']).ngroups} nodes in {dams.basin.nunique()} basins; "
          f"area ratio in [{LO},{HI}]: all {r.between(LO, HI).mean():.2f}, > 100 MCM {r[big].between(LO, HI).mean():.2f}; "
          f"capacity {dams.CAP_MCM.sum():,.0f} MCM ({dams.CAP_MCM[~r.between(LO, HI)].sum():,.0f} outside the window)")
    v3 = node_attributes(st, dams)
    v3.to_parquet(DATA / "attributes/static_v3.parquet")
    subprocess.run([sys.executable, str(HERE / "5_graph_attributes.py"), "--attributes", str(DATA / "attributes/static_v3.parquet"),
                    "--graph", str(DATA / "metadata/all_graphs.json"),
                    "--out", str(DATA / "attributes/static_graph_v3.parquet")], check=True)


if __name__ == "__main__":
    main()
