#!/bin/bash
#SBATCH --job-name=flipped_extract_sub_basins
#SBATCH --output=%j.out
#SBATCH -t 7-00:00:00
#SBATCH -p ceewater_cjgleason-cpu
#SBATCH -q long
#SBATCH -A pi_cjgleason_umass_edu
#SBATCH -c 1
#SBATCH --mem=32G

cd /nas/cee-water/cjgleason/ted/swot-ml\
source .venv/bin/activate


#######################################
# Argument parsing
#######################################
usage() {
    echo "Usage: sbatch $0 --basin-dir <path> [--save-dir <path>] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]"
    exit 1
}

# Defaults
START_DATE="1980-01-01"
END_DATE="2024-12-31"
N_WORKERS=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --basin-dir)
            BASIN_DIR="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --num_workers)
            N_WORKERS="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$BASIN_DIR" ]]; then
    echo "ERROR: --basin-dir is required."
    usage
fi


#######################################
# Job parameters
#######################################
BATCH_SIZE=16

echo "ERA5 batch processing subbasins"
echo "Start time: $(date)"
echo "Running on node: $SLURM_NODELIST"



#######################################
# Run Python
#######################################
echo "=== Starting Python processing ==="
python /nas/cee-water/cjgleason/ted/swot-ml/notebooks/reservoirs/preprocess/era5/flipped_extract_sub_basins.py \
    --basin-dir "$BASIN_DIR" \
    --save-dir "$SAVE_DIR" \
    --n-workers $N_WORKERS \
    --batch-size $BATCH_SIZE \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

# Check exit status
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "=== Job completed successfully ==="
    echo "Completion time: $(date)"
    
else
    echo "=== Job failed ==="
    echo "Failure time: $(date)"
    echo "Python exit code: $PYTHON_EXIT_CODE"
    
    exit $PYTHON_EXIT_CODE
fi

echo "Job finished at $(date)"