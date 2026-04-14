#!/bin/bash
# =============================================================================
# Con-CDVAE: Full Training Pipeline for Dual Conditions (Formation Energy + Band Gap)
# =============================================================================
# This script runs the complete two-step training pipeline:
#   Step 1: Train the CDVAE model with both formation energy and band gap conditions
#   Step 2: Train the Prior model on top of the frozen CDVAE
#
# Usage:
#   bash scripts/train_format_gap.sh [DATASET] [EXPNAME] [NGPU]
#
# Arguments:
#   DATASET  - Data config name (default: mp20_format_gap)
#   EXPNAME  - Experiment name (default: format_gap_dual)
#   NGPU     - Number of GPUs (default: 1)
#
# Examples:
#   # Quick test with sample data (single GPU):
#   bash scripts/train_format_gap.sh mptest_format_gap test_dual 1
#
#   # Full training with mp_20 dataset (single GPU):
#   bash scripts/train_format_gap.sh mp20_format_gap format_gap_dual 1
#
#   # Full training with 4 GPUs:
#   bash scripts/train_format_gap.sh mp20_format_gap format_gap_dual 4
# =============================================================================

set -e  # Exit on any error

# Parse arguments with defaults
DATASET=${1:-mp20_format_gap}
EXPNAME=${2:-format_gap_dual}
NGPU=${3:-1}

echo "============================================="
echo "Con-CDVAE Dual-Condition Training Pipeline"
echo "============================================="
echo "Dataset:     ${DATASET}"
echo "Experiment:  ${EXPNAME}"
echo "GPUs:        ${NGPU}"
echo "============================================="

# Set up environment variables
if [ -f .env ]; then
    source .env
fi

# Ensure PROJECT_ROOT is set
if [ -z "$PROJECT_ROOT" ]; then
    export PROJECT_ROOT=$(pwd)
    export HYDRA_JOBS=$(pwd)/output/hydra
    export WABDB_DIR=$(pwd)/output/wandb
    echo "Set PROJECT_ROOT=${PROJECT_ROOT}"
fi

# Create output directories
mkdir -p ${HYDRA_JOBS}
mkdir -p ${WABDB_DIR}

# =============================================================================
# STEP 1: Train CDVAE Model with Dual Conditions
# =============================================================================
echo ""
echo "============================================="
echo "STEP 1: Training CDVAE with dual conditions"
echo "  model=vae_mp_format_gap"
echo "  data=${DATASET}"
echo "============================================="

if [ "$NGPU" -gt 1 ]; then
    echo "Multi-GPU training with ${NGPU} GPUs (DDP)"
    torchrun --nproc_per_node ${NGPU} concdvae/run.py \
        data=${DATASET} \
        expname=${EXPNAME} \
        model=vae_mp_format_gap \
        train.pl_trainer.accelerator=gpu \
        train.pl_trainer.devices=${NGPU} \
        train.pl_trainer.strategy=ddp_find_unused_parameters_true
else
    echo "Single GPU training"
    python concdvae/run.py \
        data=${DATASET} \
        expname=${EXPNAME} \
        model=vae_mp_format_gap
fi

echo ""
echo "STEP 1 COMPLETE! Model checkpoint saved to:"
echo "  ${HYDRA_JOBS}/singlerun/$(date +%Y-%m-%d)/${EXPNAME}/"
echo ""

# =============================================================================
# STEP 2: Train Prior Model
# =============================================================================
# Find the latest model directory and checkpoint
MODEL_DATE=$(ls -t ${HYDRA_JOBS}/singlerun/ | head -1)
MODEL_DIR="${HYDRA_JOBS}/singlerun/${MODEL_DATE}/${EXPNAME}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found at ${MODEL_DIR}"
    echo "Please set MODEL_DIR manually and run step 2 separately."
    exit 1
fi

echo "============================================="
echo "STEP 2: Training Prior Model"
echo "  model_path=${MODEL_DIR}"
echo "============================================="

# Train default prior (uses same condition model as the CDVAE)
python concdvae/run_prior.py \
    --model_path ${MODEL_DIR} \
    --prior_label prior_default

echo ""
echo "STEP 2 (default prior) COMPLETE!"
echo ""

# Train full prior with additional conditions (optional)
echo "============================================="
echo "STEP 2b: Training Full Prior (with extra conditions)"
echo "============================================="

python concdvae/run_prior.py \
    --model_path ${MODEL_DIR} \
    --prior_label prior_full \
    --priorcondition_file mp_full \
    --data_file mptest4conz

echo ""
echo "============================================="
echo "ALL TRAINING COMPLETE!"
echo "============================================="
echo ""
echo "Model directory: ${MODEL_DIR}"
echo ""
echo "To generate crystals with both conditions, run:"
echo "  python scripts/gen_crystal.py --config conf/gen/format_gap_default.yaml"
echo ""
echo "But first update conf/gen/format_gap_default.yaml:"
echo "  model_path: ${MODEL_DIR}"
echo ""
