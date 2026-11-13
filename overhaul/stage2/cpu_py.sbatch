#!/bin/bash
#SBATCH -A swot-ml
#SBATCH -p mb
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /project/swot-ml/swot-ml/overhaul/logs/%x_%j.out
# Small CPU job for the SWOT river scripts of this worktree. Usage: sbatch -J <name> cpu_py.sbatch <script.py> <args>
cd /project/swot-ml/swot-ml && source .venv/bin/activate
S=$1; shift
python /project/swot-ml/swot-ml/.claude/worktrees/swot-river-rank/overhaul/stage2/$S "$@"
