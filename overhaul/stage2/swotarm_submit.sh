#!/bin/bash
# Experiment W (EVAL_SPEC_SWOT_RIVER.md, 2026-11-12): SWOT correlation terms, arms x seeds 0, 1 on fold 0 (c2wva settings).
# Usage: swotarm_submit.sh "<arm> <arm> ..."   arms: WR (river term), WS (reservoir term), WRS (both)
# Prints the job ids; scoring is chained by the caller (ldiag_itp.sbatch, swot_local.sbatch).
W=/project/swot-ml/swot-ml/.claude/worktrees/swot-river-rank/overhaul/stage2
R=/project/swot-ml/swot-ml/overhaul/stage2/runs
BASE="--runoff=/project/swot-ml/data/overhaul/stage2/runoff_s1s_ens --folds=lists_topo_qc --route_init=/project/swot-ml/data/overhaul/stage2/route_init_s1s.npz --loss=1 --res=1 --loss_form=l2 --lam_vol=1 --res_evap=1 --res_area=max --fold=0 --minutes=360 --eval_every=200"
for arm in $1; do
  case $arm in
    WR) X="--rsw=1";;
    WS) X="--swot=1 --swot_form=corr";;
    WRS) X="--rsw=1 --swot=1 --swot_form=corr";;
    *) echo "unknown arm $arm"; exit 1;;
  esac
  for s in 0 1; do
    suf=""; [ $s = 1 ] && suf="_s1"
    j=$(sbatch --parsable -J w_${arm}${suf} $W/run_s2w.sbatch --out=$R/s1s_f0_c2wva${arm}${suf} $BASE $X --seed=$s)
    echo "w_${arm}${suf} $j"
  done
done
