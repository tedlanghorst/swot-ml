#!/bin/bash
# Wrapper script to submit SLURM job with dynamic output path based on named argument
# Usage:
#   ./run.sh <partition> <python args...>

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <partition> <python CLI arguments>"
    exit 1
fi

partition_name="$1"
shift
python_args=("$@")  # Everything after the partition name

# Roughly validate method and path args.
method="${python_args[0]}"
path_arg="${python_args[1]}"

case "$method" in
    train|train-ensemble|grid_search|smac-optimize)
        if [[ ! -f "$path_arg" ]]; then
            echo "Error: Expected a file path for '$method', but got: $path_arg"
            exit 1
        fi
        ;;
    continue-train|test|predict|attribute|add-assimilator)
        if [[ ! -d "$path_arg" ]]; then
            echo "Error: Expected a directory path for '$method', but got: $path_arg"
            exit 1
        fi
        ;;
    *)
        echo "Warning: Unknown method '$method'. Skipping path validation."
        ;;
esac


# Set the partition and runtime args based on partition_name
n_workers=12
SBATCH_DIRECTIVES=""
ENVIRONMENT_LINES=""
case $partition_name in
    cpu)
        SBATCH_DIRECTIVES+="#SBATCH -c $((n_workers+1))\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 1-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p cpu\n"
        ENVIRONMENT_LINES+="export JAX_PLATFORMS=cpu\n"
        ;;
    ceewater)
        SBATCH_DIRECTIVES+="#SBATCH -c $((n_workers+1))\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 7-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p ceewater_kandread-cpu\n"
        ENVIRONMENT_LINES+="export JAX_PLATFORMS=cpu\n"
        ;;
    gpu)
        SBATCH_DIRECTIVES+="#SBATCH -c $n_workers\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 1-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p gpu\n"
        SBATCH_DIRECTIVES+="#SBATCH --gpus=1\n"
        SBATCH_DIRECTIVES+="#SBATCH --constraint=sm_89&vram40\n"
        ENVIRONMENT_LINES+="module load cuda/12.6\n"
        ENVIRONMENT_LINES+="export XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
        ENVIRONMENT_LINES+="export XLA_FLAGS=--xla_gpu_cuda_data_dir=\$CUDA_HOME\n"
        ENVIRONMENT_LINES+="nvidia-smi -L\n"
        ;;
    gpu-long)
        SBATCH_DIRECTIVES+="#SBATCH -c $n_workers\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 14-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p gpu\n"
        SBATCH_DIRECTIVES+="#SBATCH -q long\n"
        SBATCH_DIRECTIVES+="#SBATCH --gpus=1\n"
        SBATCH_DIRECTIVES+="#SBATCH --constraint=sm_89&vram40\n"
        ENVIRONMENT_LINES+="module load cuda/12.6\n"
        ENVIRONMENT_LINES+="export XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
        ENVIRONMENT_LINES+="export XLA_FLAGS=--xla_gpu_cuda_data_dir=\$CUDA_HOME\n"
        ENVIRONMENT_LINES+="nvidia-smi -L\n"
        ;;
    gpupod)
        SBATCH_DIRECTIVES+="#SBATCH -c $n_workers\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 14-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p gpupod-l40s\n"
        SBATCH_DIRECTIVES+="#SBATCH -q gpu-quota-16\n"
        SBATCH_DIRECTIVES+="#SBATCH -A pi_cjgleason_umass_edu\n"
        SBATCH_DIRECTIVES+="#SBATCH --nodes=1\n"
        SBATCH_DIRECTIVES+="#SBATCH --gpus=l40s:1\n"
        ENVIRONMENT_LINES+="module load cuda/12.6\n"
        ENVIRONMENT_LINES+="export XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
        ENVIRONMENT_LINES+="export XLA_FLAGS=--xla_gpu_cuda_data_dir=\$CUDA_HOME\n"
        ENVIRONMENT_LINES+="nvidia-smi -L\n"
        ;;
    *)
        echo "Unknown partition type: $partition_name"
        exit 1
        ;;
esac

# Determine job name components from the config path
config_abs_path=$(realpath "$path_arg")
config_dir=$(dirname "$config_abs_path")

# expecting runs/<experiment>/<config> where they will be many configs per experiment.
config_name=$(basename "$config_abs_path" .yml)
experiment_name=$(basename "$config_dir")

# Output directory: e.g., runs/experiment/_slurm_outputs
output_dir="${config_dir}/_slurm_outputs"
mkdir -p "$output_dir"

# Create the SBATCH script with dynamic output path and partition
sbatch_script=$(mktemp)
cat << EOF > "$sbatch_script"
#!/bin/bash
#SBATCH --job-name="${method}_${experiment_name}_${config_name}"
#SBATCH --mem=96G  # Requested Memory
#SBATCH -o ${output_dir}/${config_name}_%j.out
$(echo -e "$SBATCH_DIRECTIVES")

source .venv/bin/activate
$(echo -e "$ENVIRONMENT_LINES")

cd "$(dirname "$0")/../src"
python run.py ${python_args[*]}
EOF

# Submit the job.
sbatch "$sbatch_script"

# # View the job script.
# cat $sbatch_script

