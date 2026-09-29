"""
Adds graph-consistent static attributes to the attributes table.

The HydroATLAS attributes in static.parquet were joined from level-09 HydroBASINS polygons
near each subbasin's centroid. For about a third of gauges the matched polygon's upstream
area differs from the node's own by more than 50%, so pour-point attributes (dis_m3_*,
dor_pc_pva) can describe a different river. This script derives replacements by
accumulating local attributes down the river graph, which is consistent with uparea_km2 by
construction:

  gu_<attr>   area-weighted mean of a local attribute over the node's whole upstream area
  gu_q_m3s    mean annual flow from accumulated local runoff (run_mm_syr * area_km2)
  gu_cap_mcm  total reservoir capacity upstream (GDW CAP_MCM)
  gu_n_dams   number of GDW dams upstream
  gu_dor_pc   degree of regulation: upstream capacity as % of mean annual flow volume
  dam_local   1.0 if the node itself contains a GDW dam

Usage:
    python 5_graph_attributes.py [--attributes ...] [--graph ...] [--out ...]
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path("/project/swot-ml/swot-ml/data/multigraph_manual")

# Local (sub-basin level) attributes averaged over the upstream area.
MEAN_ATTRS = [
    "run_mm_syr", "pre_mm_syr", "tmp_dc_syr", "pet_mm_syr", "aet_mm_syr", "ari_ix_sav",
    "snw_pc_syr", "snw_pc_smx", "ele_mt_sav", "slp_dg_sav", "sgr_dk_sav", "gwt_cm_sav",
    "cly_pc_sav", "slt_pc_sav", "snd_pc_sav", "soc_th_sav", "swc_pc_syr", "kar_pc_sse",
    "for_pc_sse", "crp_pc_sse", "pst_pc_sse", "ire_pc_sse", "urb_pc_sse", "gla_pc_sse",
    "prm_pc_sse", "lka_pc_sse", "wet_pc_sg1", "wet_pc_sg2", "inu_pc_slt", "ppd_pk_sav",
    "hft_ix_s09",
]
SECONDS_PER_YEAR = 365.25 * 86400


def accumulate(graph: nx.DiGraph, order: list, values: np.ndarray, index: dict) -> np.ndarray:
    """Sums per-node values over each node's upstream area (edges point downstream)."""
    acc = values.copy()
    for node in order:
        i = index[node]
        for downstream in graph.successors(node):
            acc[index[downstream]] += acc[i]
    return acc


def graph_attributes(basin_df: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
    nodes = list(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    order = list(nx.topological_sort(graph))
    df = basin_df.reindex(nodes)

    area = df["area_km2"].fillna(0.0).to_numpy(dtype=np.float64)
    out = pd.DataFrame(index=pd.Index(nodes, name="subbasin"))

    # Area-weighted upstream means, ignoring nodes where the attribute is missing.
    attrs = [a for a in MEAN_ATTRS if a in df.columns]
    x = df[attrs].to_numpy(dtype=np.float64)
    valid = np.isfinite(x)
    weighted_sum = accumulate(graph, order, np.where(valid, x, 0.0) * area[:, None], index)
    weight = accumulate(graph, order, valid * area[:, None], index)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(weight > 0, weighted_sum / weight, np.nan)
    for j, a in enumerate(attrs):
        out[f"gu_{a}"] = means[:, j]

    # Mean annual flow from accumulated runoff depth: mm * km2 = 1e3 m3.
    runoff_m3_yr = df["run_mm_syr"].fillna(0.0).to_numpy(dtype=np.float64) * area * 1e3
    flow_m3_yr = accumulate(graph, order, runoff_m3_yr, index)
    out["gu_q_m3s"] = flow_m3_yr / SECONDS_PER_YEAR

    has_dam = df["has_dam"].fillna(False).astype(float).to_numpy()
    capacity = df["CAP_MCM"].fillna(0.0).to_numpy(dtype=np.float64) * has_dam
    out["gu_cap_mcm"] = accumulate(graph, order, capacity, index)
    out["gu_n_dams"] = accumulate(graph, order, has_dam, index)
    with np.errstate(invalid="ignore", divide="ignore"):
        dor = 100.0 * out["gu_cap_mcm"].to_numpy() * 1e6 / flow_m3_yr
    out["gu_dor_pc"] = np.where(flow_m3_yr > 0, dor, 0.0)
    out["dam_local"] = has_dam
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", type=Path, default=DATA_DIR / "attributes" / "static.parquet")
    parser.add_argument("--graph", type=Path, default=DATA_DIR / "metadata" / "all_graphs.json")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "attributes" / "static_graph.parquet")
    args = parser.parse_args()

    static = pd.read_parquet(args.attributes)
    with open(args.graph) as f:
        G = nx.readwrite.json_graph.node_link_graph(json.load(f), edges="edges")

    parts = []
    for nodes in nx.weakly_connected_components(G):
        sub = G.subgraph(nodes)
        outlets = [n for n in sub.nodes if sub.out_degree(n) == 0]
        if len(outlets) != 1:
            raise ValueError(f"Expected exactly one outlet per basin, found {outlets}")
        basin = outlets[0]
        if basin not in static.index.get_level_values("basin"):
            continue
        basin_df = static.xs(basin, level="basin")
        derived = graph_attributes(basin_df, sub)
        derived["basin"] = basin
        parts.append(derived.reset_index().set_index(["basin", "subbasin"]))

    derived = pd.concat(parts)
    new_cols = list(derived.columns)
    overlap = static.columns.intersection(new_cols)
    if len(overlap):
        raise ValueError(f"Derived columns already exist in {args.attributes}: {list(overlap)}")
    combined = static.join(derived, how="left")
    missing = combined[new_cols[0]].isna().sum()
    combined.to_parquet(args.out)
    print(f"Wrote {args.out}: {len(combined)} rows, {len(new_cols)} new columns "
          f"({missing} rows not in any graph).")


if __name__ == "__main__":
    main()
