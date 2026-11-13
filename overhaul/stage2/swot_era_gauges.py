"""Gauge skill in the SWOT period (2023-07-01 .. end of record) from swot_era_flows.py output (2026-11-12). Reported
only: these gauge days are the test window of EVAL_SPEC (never used in training or selection), and nothing is chosen
on this table. Per gauge with >= 180 observed days: NSE, |log beta|, Jul-Sep volume ratio, per run (full); paired
arm - control (seed means) by class (stage2/diag/gauge_classes.parquet), median with a gauge-bootstrap CI.
Usage: python swot_era_gauges.py --diag=DIR --control=c2wva,c2wva_s1 --arms=WR:WR,WR_s1;... -> DIR/swot_era_gauges.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
DIAG = Path(opt["diag"])
CONTROL = opt.get("control", "c2wva,c2wva_s1").split(",")
ARMS = {a.split(":")[0]: a.split(":")[1].split(",") for a in opt.get("arms", "").split(";") if a}
BNAME = {"USGS-07289000": "Mississippi", "USGS-09521100": "Colorado", "USGS-14246900": "Columbia",
         "USGS-02359170": "ACF", "USGS-01578310": "Susquehanna"}
rng = np.random.default_rng(0)
lines, rows = [], []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


for basin, bn in BNAME.items():
    z = np.load(DIAG / f"flows_{basin}.npz", allow_pickle=True)
    isg = z["is_gauge"]
    names, sets = z["nodes"].astype(str)[isg], z["sets"].astype(str)[isg]
    obs = z["obs"]
    jas = pd.DatetimeIndex(z["dates"]).month.isin([7, 8, 9])
    runs = [k[:-6] for k in z.files if k.endswith("__full")]
    for j, g in enumerate(names):
        ok = np.isfinite(obs[:, j])
        if ok.sum() < 180:
            continue
        y = obs[ok, j].astype(float)
        row = {"basin": bn, "gauge": g, "set": sets[j]}
        for r in runs:
            x = z[f"{r}__full"][:, isg][ok, j].astype(float)
            row[f"nse__{r}"] = 1 - ((x - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            row[f"alb__{r}"] = abs(np.log(max(x.mean(), 1e-9) / max(y.mean(), 1e-9)))
        rows.append(row)
G = pd.DataFrame(rows)
C = pd.read_parquet("/project/swot-ml/swot-ml/overhaul/stage2/diag/gauge_classes.parquet").drop_duplicates("gauge").set_index("gauge")
G = G.join(C[["depleted_0.9", "below_res", "regmain"]], on="gauge")
G.to_parquet(DIAG / "swot_era_gauges.parquet", index=False)
classes = {"all": G.basin.notna(), "depleted < 0.9": G["depleted_0.9"] == 1, "below reservoirs": G.below_res == True,  # noqa: E712
           "regulated main stem": G.regmain == True, "test-fold gauges": G.set == "test"}  # noqa: E712
say(f"gauge skill 2023-07+ (test window; reported only): {len(G)} gauges with >= 180 observed days")
ctl = G[[f"nse__{r}" for r in CONTROL]].mean(1)
say("control median NSE by class: " + str({k: (round(ctl[m].median(), 3), int(m.sum())) for k, m in classes.items()}))
for a, names in ARMS.items():
    if not all(f"nse__{n}" in G.columns for n in names):
        continue
    say(f"\narm {a} - control (seed means), median paired change in NSE [95% CI], and median |log beta| arm vs control:")
    d = G[[f"nse__{n}" for n in names]].mean(1) - ctl
    for k, m in classes.items():
        v = d[m].dropna().values
        if len(v) < 5:
            continue
        bs = np.median(v[rng.integers(0, len(v), (1000, len(v)))], 1)
        say(f"  {k:20s} {np.median(v):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (n {len(v)}, {np.mean(v > 0):.0%} up); "
            f"|log beta| {G.loc[m, [f'alb__{n}' for n in names]].mean(1).median():.3f} vs {G.loc[m, [f'alb__{r}' for r in CONTROL]].mean(1).median():.3f}; "
            f"by basin {d[m].groupby(G.basin[m]).median().round(3).to_dict()}")
(DIAG / "swot_era_gauges.txt").write_text("\n".join(lines) + "\n")
