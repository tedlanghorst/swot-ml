#!/bin/bash
#SBATCH -A swot-ml
#SBATCH -p mb
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /project/swot-ml/swot-ml/overhaul/logs/%x_%j.out
# SWORD reaches -> overhaul graph nodes + filtered RiverSP reach observations (2026-11-12, CPU).
cd /project/swot-ml/swot-ml && source .venv/bin/activate
python /project/swot-ml/swot-ml/.claude/worktrees/swot-river-rank/overhaul/data_prep/swot_reach_match.py
