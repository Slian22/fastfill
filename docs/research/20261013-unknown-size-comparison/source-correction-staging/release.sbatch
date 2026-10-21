#!/bin/bash
#SBATCH --job-name=source-correction-release
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=none
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=source-correction-release-%j.out
#SBATCH --error=source-correction-release-%j.err

set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: sbatch release.sbatch /abs/config.json /abs/run_release.py /abs/python' >&2
    exit 64
fi
for task_path in "$@"; do
    if [[ "$task_path" != /* || ! -f "$task_path" ]]; then
        echo 'All three arguments must be existing absolute file paths.' >&2
        exit 64
    fi
done
if [[ ! -x "$3" ]]; then
    echo 'Python executable is not executable.' >&2
    exit 64
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# timeout controls the entire wrapper + child process group, including audits.
# Slurm's 2h cap provides cleanup margin. No retries, requeues, GPU or refit.
exec timeout --signal=TERM --kill-after=60s 115m "$3" -I -B -u "$2" --config "$1"
