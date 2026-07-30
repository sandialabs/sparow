#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# User settings
# ==========================================================

# mpi-sppy-compatible module for the Advanced Farmers problem.
MODULE_NAME="sparow_examples.farmers.bootsp_advanced_farmers"

SOLVER="${1:-gurobi_direct}"

# ----------------------------------------------------------
# Precomputed candidate solution xhat
# ----------------------------------------------------------
XHAT_FNAME="bootsp_farmer_candidate_xhat.npy"

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
MAX_COUNT=1000                 # total number of available population scenarios
CANDIDATE_SAMPLE_SIZE=0       
SAMPLE_SIZE=500                
SUBSAMPLE_SIZE=100             # used only for subsampling / bagging methods
NB=10                          # number of bootstrap/bagging resamples
ALPHA=0.05                     # significance level
SEED_OFFSET=678                

# Optional: model-specific flags can be appended here if needed
EXTRA_ARGS=""

METHODS=(
  "Classical_gaussian"
#  "Classical_quantile"
#   "Extended"
#   "Subsampling"
#   "Bagging_with_replacement"
#   "Bagging_without_replacement"
)

echo "============================================================"
echo "Running boot-sp methods on Advanced Farmers with fixed xhat"
echo "Module: ${MODULE_NAME}"
echo "Solver: ${SOLVER}"
echo "xhat file: ${XHAT_FNAME}"
echo "max_count: ${MAX_COUNT}"
echo "candidate_sample_size: ${CANDIDATE_SAMPLE_SIZE}"
echo "sample_size: ${SAMPLE_SIZE}"
echo "subsample_size: ${SUBSAMPLE_SIZE}"
echo "nB: ${NB}"
echo "alpha: ${ALPHA}"
echo "seed_offset: ${SEED_OFFSET}"
echo "============================================================"
echo

for METHOD in "${METHODS[@]}"; do
    echo "------------------------------------------------------------"
    echo "Running boot-sp method: ${METHOD}"
    echo "------------------------------------------------------------"

    python -m mpisppy.confidence_intervals.bootsp.user_boot "${MODULE_NAME}" \
        --max-count "${MAX_COUNT}" \
        --candidate-sample-size "${CANDIDATE_SAMPLE_SIZE}" \
        --sample-size "${SAMPLE_SIZE}" \
        --subsample-size "${SUBSAMPLE_SIZE}" \
        --nB "${NB}" \
        --alpha "${ALPHA}" \
        --seed-offset "${SEED_OFFSET}" \
        --solver-name "${SOLVER}" \
        --xhat-fname "${XHAT_FNAME}" \
        --boot-method "${METHOD}" \
        ${EXTRA_ARGS}

    echo
done

echo "Finished running all boot-sp methods."