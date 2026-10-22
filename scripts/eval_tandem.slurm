#!/bin/bash
#SBATCH --job-name=tandem-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log

# Evaluate Tandem-DETR weights on VisDrone val at 800x800 (COCO metrics,
# maxDets=100). Submit from the repository root, with your Python environment
# active and `logs/` created.
#
#   sbatch --export=ALL,SIZE=m,CKPT=/path/to/tandem_detr_m_visdrone.pth scripts/eval_tandem.slurm

set -euo pipefail

SIZE="${SIZE:?set SIZE=m, l or x}"
case "${SIZE}" in
    m|l|x) ;;
    *) echo "SIZE must be m, l or x" >&2; exit 2 ;;
esac
CKPT="${CKPT:?set CKPT=/path/to/weights.pth}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/output/eval/tandem_${SIZE}_${SLURM_JOB_ID:-local}}"
MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1

echo "Config: configs/tandem/tandem_${SIZE}.yml   Weights: ${CKPT}"
torchrun --master_port="${MASTER_PORT}" --nproc_per_node=1 train.py \
    -c "configs/tandem/tandem_${SIZE}.yml" \
    --test-only -r "${CKPT}" \
    --seed 0 \
    --output-dir "${OUT_DIR}"
