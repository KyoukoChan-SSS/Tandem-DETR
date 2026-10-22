#!/bin/bash
#SBATCH --job-name=tandem-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80g
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.log

# Train Tandem-DETR-{M,L,X} on VisDrone at 800x800 (recipe fully in the config).
# Submit from the repository root, with your Python environment active and
# `logs/` created; add the partition / account flags your cluster needs.
#
#   mkdir -p logs
#   sbatch --export=ALL,SIZE=m scripts/train_tandem.slurm
#
# Self-resuming: the run dir is fixed per SIZE (and optional TAG). A job that
# finds last.pth resumes from it; a job that finds DONE exits. Chain jobs with
# --dependency=afterany:<prev> to go past the walltime limit.

set -euo pipefail

SIZE="${SIZE:?set SIZE=m, l or x}"
case "${SIZE}" in
    m|l|x) ;;
    *) echo "SIZE must be m, l or x" >&2; exit 2 ;;
esac
TAG="${TAG:-}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="configs/tandem/tandem_${SIZE}.yml"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/output}"
RUN_DIR="${RUN_ROOT}/tandem_${SIZE}${TAG:+_${TAG}}"
MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"

cd "${REPO_DIR}"
mkdir -p "${RUN_DIR}/tensorboard"
if [ -f "${RUN_DIR}/DONE" ]; then
    echo "${RUN_DIR}/DONE exists, nothing to do."
    exit 0
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME=()
if [ -f "${RUN_DIR}/last.pth" ]; then
    RESUME=(-r "${RUN_DIR}/last.pth")
fi

echo "============================================"
echo "Start:   $(date '+%F %T %Z')  Host: $(hostname)  Job: ${SLURM_JOB_ID:-n/a}"
echo "Commit:  $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Config:  ${CONFIG}"
echo "Run dir: ${RUN_DIR}"
echo "Resume:  ${RESUME[*]:-<from scratch>}"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cp -n "${CONFIG}" "${RUN_DIR}/" 2>/dev/null || true

torchrun --master_port="${MASTER_PORT}" --nproc_per_node=1 train.py \
    -c "${CONFIG}" \
    --seed 0 \
    --output-dir "${RUN_DIR}" \
    --summary-dir "${RUN_DIR}/tensorboard" \
    ${RESUME[@]+"${RESUME[@]}"}

touch "${RUN_DIR}/DONE"
echo "End: $(date '+%F %T %Z')"
