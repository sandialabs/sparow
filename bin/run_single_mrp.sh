#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# User settings
# ==========================================================

MODEL_MODULE="sparow_examples.farmers.MRPfarmers"
MODEL_NAME="Advanced"
LF_MODEL_TYPE="classic"
SOLVER="gurobi_direct"
VERBOSE="true"

# Candidate-solution generation (if you don't want to use existing file)
CANDIDATE_SCEN_COUNT=5
CANDIDATE_SEED=12345
CANDIDATE_WITH_REPLACEMENT="false"

# If you want to use existing candidate solution, already written to file
USE_EXISTING_XHAT="false"

# MRP confidence interval settings
ALPHA=0.05
MRP_SEED=678
MRP_WITH_REPLACEMENT="true"

# Single-run settings
XHAT_FILE="farmer_candidate_xhat_cand${CANDIDATE_SCEN_COUNT}_seed${CANDIDATE_SEED}.npy"
N=500
M=10
COMPUTE_TRUE_GAP="true"

# ==========================================================
# Convert replacement choice into CLI flag
# ==========================================================

VERBOSE_FLAG=""
if [ "${VERBOSE}" = "true" ]; then
    VERBOSE_FLAG="--verbose"
elif [ "${VERBOSE}" = "false" ]; then
    VERBOSE_FLAG=""
else
    echo "Error: VERBOSE must be 'true' or 'false'"
    exit 1
fi

USE_EXISTING_XHAT_FLAG=""
if [ "${USE_EXISTING_XHAT}" = "true" ]; then
    USE_EXISTING_XHAT_FLAG="--use-existing-xhat"
elif [ "${USE_EXISTING_XHAT}" = "false" ]; then
    USE_EXISTING_XHAT_FLAG=""
else
    echo "Error: USE_EXISTING_XHAT must be 'true' or 'false'"
    exit 1
fi

CANDIDATE_FLAG=""
if [ "${CANDIDATE_WITH_REPLACEMENT}" = "true" ]; then
    CANDIDATE_FLAG="--candidate-with-replacement"
elif [ "${CANDIDATE_WITH_REPLACEMENT}" = "false" ]; then
    CANDIDATE_FLAG="--candidate-without-replacement"
else
    echo "Error: CANDIDATE_WITH_REPLACEMENT must be 'true' or 'false'"
    exit 1
fi

MRP_FLAG=""
if [ "${MRP_WITH_REPLACEMENT}" = "true" ]; then
    MRP_FLAG="--mrp-with-replacement"
elif [ "${MRP_WITH_REPLACEMENT}" = "false" ]; then
    MRP_FLAG="--mrp-without-replacement"
else
    echo "Error: MRP_WITH_REPLACEMENT must be 'true' or 'false'"
    exit 1
fi

TRUE_GAP_FLAG=""
if [ "${COMPUTE_TRUE_GAP}" = "true" ]; then
    TRUE_GAP_FLAG="--compute-true-gap"
elif [ "${COMPUTE_TRUE_GAP}" = "false" ]; then
    TRUE_GAP_FLAG=""
else
    echo "Error: COMPUTE_TRUE_GAP must be 'true' or 'false'"
    exit 1
fi

echo "=== Running single standard MRP experiment ==="
echo "Candidate sampling flag: ${CANDIDATE_FLAG}"
echo "MRP sampling flag: ${MRP_FLAG}"
echo "xhat file: ${XHAT_FILE}"
echo "candidate_scen_count: ${CANDIDATE_SCEN_COUNT}"
echo "n: ${N}"
echo "m: ${M}"

python -m sparow.conf_intervals.cli \
    --model-module "${MODEL_MODULE}" \
    --model-name "${MODEL_NAME}" \
    --lf-model-type "${LF_MODEL_TYPE}" \
    --solver-name "${SOLVER}" \
     ${VERBOSE_FLAG} \
    --candidate-scen-count "${CANDIDATE_SCEN_COUNT}" \
    --candidate-seed "${CANDIDATE_SEED}" \
    ${CANDIDATE_FLAG} \
    ${USE_EXISTING_XHAT_FLAG} \
    --alpha "${ALPHA}" \
    --mrp-seed "${MRP_SEED}" \
    ${MRP_FLAG} \
    --xhat-file "${XHAT_FILE}" \
    --n "${N}" \
    --m "${M}" \
    ${TRUE_GAP_FLAG}

echo "Single MRP experiment complete."