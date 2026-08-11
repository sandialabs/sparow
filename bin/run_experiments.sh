#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# User settings
# ==========================================================

MODEL_MODULE="sparow_examples.mrp_facilityloc.mrp_discrete_facilityloc"
MODEL_NAME="HF"
LF_MODEL_TYPE="classic"
SOLVER="gurobi_direct"
VERBOSE="true"
ACV_MRP="true" # Specify whether to run ACV-MRP or standard MRP. If true, will run ACV-MRP.

# Candidate-solution generation (if you don't want to use existing file)
CANDIDATE_SCEN_COUNT=0
CANDIDATE_SEED=0
CANDIDATE_WITH_REPLACEMENT="false"

# If you want to use existing candidate solution, already written to file
USE_EXISTING_XHAT="true"

# Confidence interval settings
ALPHA=0.05
MRP_SEED=678
MRP_WITH_REPLACEMENT="true"

# Grid of m and n values, optionally M values
M_VALUES="10,20"
N_VALUES="200,100"
ACV_M_VALUES="5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Filepaths - should be in same folder as the script where your problem instance is defined
XHAT_FILE="${SCRIPT_DIR}/../../sparow_examples/sparow_examples/mrp_facilityloc/manually_created_suboptimal_xhat.npy"
RESULTS_CSV="grid_results.csv" # this file gets automatically written to correct folder
CLI_LOG="${SCRIPT_DIR}/../../sparow_examples/sparow_examples/mrp_facilityloc/cli_run.log"

# Filepaths for scripts containing plotting functionality
PLOT_SCRIPT_STANDARD="${SCRIPT_DIR}/../sparow/conf_intervals/plot_results/plot_mrp_results.py"
PLOT_SCRIPT_ACV="${SCRIPT_DIR}/../sparow/conf_intervals/plot_results/plot_acvmrp_results.py"

# ==========================================================
# Convert replacement choices into CLI flags
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

ACV_FLAG=""
if [ "${ACV_MRP}" = "true" ]; then
    ACV_FLAG="--acv-mrp"
elif [ "${ACV_MRP}" = "false" ]; then
    ACV_FLAG=""
else
    echo "Error: ACV_MRP must be 'true' or 'false'"
    exit 1
fi

echo "=== Running grid experiments ==="
echo "Candidate sampling flag: ${CANDIDATE_FLAG}"
echo "MRP sampling flag: ${MRP_FLAG}"
echo "ACV flag: ${ACV_FLAG}"
echo "Use existing xhat flag: ${USE_EXISTING_XHAT_FLAG}"

# Note that as of now, grid-experiment will always do nested sampling of the 
# scenarios, so no additional flag for --nested-sampling is needed.
# .... but may change this in future refactor

if [ "${ACV_MRP}" = "true" ]; then
    python -m sparow.conf_intervals.cli \
        --grid-experiment \
        --acv-mrp \
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
        --m-values "${M_VALUES}" \
        --n-values "${N_VALUES}" \
        --M-values "${ACV_M_VALUES}" \
        --xhat-file "${XHAT_FILE}" \
        --output-csv "${RESULTS_CSV}" | tee "${CLI_LOG}" # writes terminal output to CLI_LOG

    ACTUAL_RESULTS_CSV=$(grep "Wrote CSV:" "${CLI_LOG}" | tail -n 1 | sed 's/.*Wrote CSV: //')
    ACTUAL_XHAT_FILE=$(grep "Wrote xhat:" "${CLI_LOG}" | tail -n 1 | sed 's/.*Wrote xhat: //')

    echo "Wrote ${ACTUAL_RESULTS_CSV}"
    echo "Wrote ${ACTUAL_XHAT_FILE}"

    python "${PLOT_SCRIPT_ACV}" "${ACTUAL_RESULTS_CSV}"
    echo "ACVMRP plots created."
else 
    python -m sparow.conf_intervals.cli \
        --grid-experiment \
        --model-module "${MODEL_MODULE}" \
        --model-name "${MODEL_NAME}" \
        --lf-model-type "${LF_MODEL_TYPE}" \
        --solver-name "${SOLVER}" \
        --candidate-scen-count "${CANDIDATE_SCEN_COUNT}" \
        --candidate-seed "${CANDIDATE_SEED}" \
        ${CANDIDATE_FLAG} \
        ${USE_EXISTING_XHAT_FLAG} \
        --alpha "${ALPHA}" \
        --mrp-seed "${MRP_SEED}" \
        ${MRP_FLAG} \
        --m-values "${M_VALUES}" \
        --n-values "${N_VALUES}" \
        --xhat-file "${XHAT_FILE}" \
        --output-csv "${RESULTS_CSV}" | tee "${CLI_LOG}" # writes terminal output to CLI_LOG

    ACTUAL_RESULTS_CSV=$(grep "Wrote CSV:" "${CLI_LOG}" | tail -n 1 | sed 's/.*Wrote CSV: //')
    ACTUAL_XHAT_FILE=$(grep "Wrote xhat:" "${CLI_LOG}" | tail -n 1 | sed 's/.*Wrote xhat: //')

    echo "Wrote ${ACTUAL_RESULTS_CSV}"
    echo "Wrote ${ACTUAL_XHAT_FILE}"

    python "${PLOT_SCRIPT_STANDARD}" "${ACTUAL_RESULTS_CSV}"
    echo "Standard MRP plots created."
fi

