#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# User settings
# ==========================================================

MODEL_MODULE="sparow_examples.farmers.MRPfarmers"
MODEL_NAME="Advanced"
SOLVER="gurobi_direct"

# Candidate-solution generation
CANDIDATE_SCEN_COUNT=5
CANDIDATE_SEED=12345
CANDIDATE_WITH_REPLACEMENT="false"

# Confidence interval settings
ALPHA=0.05
MRP_SEED=678
MRP_WITH_REPLACEMENT="true"

# Grid of m and n values, optionally M values
M_VALUES="10,20,30"
N_VALUES="625, 312, 100"
ACV_MRP="true"
ACV_M_VALUES="5,10"

# Files
XHAT_FILE="GRID_candidate_xhat_cand${CANDIDATE_SCEN_COUNT}_seed${CANDIDATE_SEED}.npy"
RESULTS_CSV="grid_results.csv"
PLOT_SCRIPT_STANDARD="plot_mrp_results.py"
PLOT_SCRIPT_ACV="plot_acvmrp_results.py"

# ==========================================================
# Convert replacement choices into CLI flags
# ==========================================================

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

ACV_FLAG=""
if [ "${ACV_MRP}" = "true" ]; then
    ACV_FLAG="--acv-mrp"
elif [ "${ACV_MRP}" = "false" ]; then
    ACV_FLAG=""
else
    echo "Error: ACV_MRP must be 'true' or 'false'"
    exit 1
fi


echo "=== Running experiments for standard MRP algorithm ==="
echo "Candidate sampling flag: ${CANDIDATE_FLAG}"
echo "MRP sampling flag: ${MRP_FLAG}"
echo "ACV flag: ${ACV_FLAG}"

# Note that for standard MRP, grid-experiment will always do nested sampling of the 
# scenarios, so no additional flag for --nested-sampling is needed.
if [ "${ACV_MRP}" = "true" ]; then
    python -m sparow.ci.cli \
        --grid-experiment \
        --acv-mrp \
        --model-module "${MODEL_MODULE}" \
        --model-name "${MODEL_NAME}" \
        --solver-name "${SOLVER}" \
        --candidate-scen-count "${CANDIDATE_SCEN_COUNT}" \
        --candidate-seed "${CANDIDATE_SEED}" \
        ${CANDIDATE_FLAG} \
        --alpha "${ALPHA}" \
        --mrp-seed "${MRP_SEED}" \
        ${MRP_FLAG} \
        --m-values "${M_VALUES}" \
        --n-values "${N_VALUES}" \
        --M-values "${ACV_M_VALUES}" \
        --xhat-file "${XHAT_FILE}" \
        --output-csv "${RESULTS_CSV}"

    echo "Wrote ${RESULTS_CSV}"
    echo "Wrote ${XHAT_FILE}"

    python "${PLOT_SCRIPT_ACV}" "${RESULTS_CSV}"
    echo "ACVMRP plots created."
else 
    python -m sparow.ci.cli \
        --grid-experiment \
        --model-module "${MODEL_MODULE}" \
        --model-name "${MODEL_NAME}" \
        --solver-name "${SOLVER}" \
        --candidate-scen-count "${CANDIDATE_SCEN_COUNT}" \
        --candidate-seed "${CANDIDATE_SEED}" \
        ${CANDIDATE_FLAG} \
        --alpha "${ALPHA}" \
        --mrp-seed "${MRP_SEED}" \
        ${MRP_FLAG} \
        --m-values "${M_VALUES}" \
        --n-values "${N_VALUES}" \
        --xhat-file "${XHAT_FILE}" \
        --output-csv "${RESULTS_CSV}"

    echo "Wrote ${RESULTS_CSV}"
    echo "Wrote ${XHAT_FILE}"

    python "${PLOT_SCRIPT_STANDARD}" "${RESULTS_CSV}"
    echo "Standard MRP plots created."
fi

