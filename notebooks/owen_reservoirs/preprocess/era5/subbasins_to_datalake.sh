#!/bin/bash
#SBATCH --job-name=extract_era5_to_datalake
#SBATCH --output=%j.out
#SBATCH -t 7-00:00:00
#SBATCH -p cpu,ceewater_cjgleason-cpu
#SBATCH -q long
#SBATCH -A pi_cjgleason_umass_edu
#SBATCH -c 1
#SBATCH --mem=64G
set -e

cd /nas/cee-water/cjgleason/ted/swot-ml
source .venv/bin/activate


python /nas/cee-water/cjgleason/ted/swot-ml/notebooks/multigraph_manual/preprocess/era5/flipped_extract_sub_basins.py \
    --basin-file /nas/cee-water/cjgleason/ted/swot-ml/data/owen_reservoirs/Susquehanna/metadata/subbbasins_4km2.parquet \
    --save-dir /nas/cee-water/cjgleason/ted/swot-ml/data/owen_reservoirs/Susquehanna/datalakes/training \
    --n-workers 48 \
    --batch-size 16 \
    --start-date "2000-01-01" \
    --end-date "2025-12-31"
