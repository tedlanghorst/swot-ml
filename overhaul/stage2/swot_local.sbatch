#!/bin/bash
#SBATCH -A swot-ml
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH --partition=inv-langhorst,mb-l40s,mb-h100
#SBATCH --gres=gpu:1
#SBATCH -o /project/swot-ml/swot-ml/overhaul/logs/%x_%j.out
# Local toggle (2026-11-12): SWOT-era flows of the fold-0 runs with the routed upstream-gauge flows (GPU, inference
# only), then swot_local.py (and swot_rank.py on the same flows). Usage: sbatch swot_local.sbatch "<arms>" [diag name]
ulimit -Su 16384 2>/dev/null || true
cd /project/swot-ml/swot-ml && source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLOSC_NTHREADS=2
nvidia-smi -L
W=/project/swot-ml/swot-ml/.claude/worktrees/swot-river-rank/overhaul/stage2
D=/project/swot-ml/swot-ml/overhaul/stage2/diag/${2:-swot_local_f0}
RUNS="b:s1s_f0_b_route:params_best,c2wva:s1s_f0_c2wva:params_best,c2wva_s1:s1s_f0_c2wva_s1:params_best"
ARMS=""
for arm in $1; do
  RUNS="$RUNS,$arm:s1s_f0_c2wva$arm:params_best,${arm}_s1:s1s_f0_c2wva${arm}_s1:params_best"
  ARMS="$ARMS;$arm:$arm,${arm}_s1"
done
python $W/swot_era_flows.py --fold=0 --local=1 --out=$D --runs="$RUNS" \
  && python $W/swot_local.py --diag=$D --control=c2wva,c2wva_s1 --arms="${ARMS#;}" \
  && python $W/swot_rank.py --diag=$D --control=c2wva,c2wva_s1 --arms="${ARMS#;}" --base=b
