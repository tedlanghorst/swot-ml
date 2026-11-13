#!/bin/bash
#SBATCH -A swot-ml
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH --partition=inv-langhorst,mb-l40s,mb-h100
#SBATCH --gres=gpu:1
#SBATCH -o /project/swot-ml/swot-ml/overhaul/logs/%x_%j.out
# train_s2.py of the worktree swot-river-rank (SWOT correlation terms). Usage: sbatch -J <name> run_s2w.sbatch <args>
ulimit -Su 16384 2>/dev/null || true
cd /project/swot-ml/swot-ml && source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLOSC_NTHREADS=2
nvidia-smi -L
python /project/swot-ml/swot-ml/.claude/worktrees/swot-river-rank/overhaul/stage2/train_s2.py "$@"
