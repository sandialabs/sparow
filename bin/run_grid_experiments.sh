#!/usr/bin/env bash
set -euo pipefail

# This script runs grid-based numerical experiments (parameter sweeps) for either single-fidelity
# standard MRP or multifidelity ACV-MRP, depending on the ACV_MRP setting.
# This script automatically writes results to CSV files, and also automatically generate plots using 
# the appropriate scripts in sparow/bin

# NOTE: this script does NOT use macro-replications, so each parameter configuration produces just one
# realized point estimate and one realized confidence interval, rather than an empirical distribution 
# of outcomes across repeated seeds.
# This matters because a 95% confidence level is a repeated-sampling statement:
# over many repeated runs, about 95 out of 100 intervals are expected to contain
# the true population quantity, so macro-replications are useful for checking this behavior in practice.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==========================================================
# User settings
# ==========================================================

# Python module that defines either: get_sp_model_for_uq(...) or get_model_ensemble_for_uq(...)
MODEL_MODULE="sparow_examples.uq_opf.uq_opf"

# Specify whether to run ACV-MRP or standard MRP. If true, will run ACV-MRP.
ACV_MRP="true" 

# Optional model name passed into either get_sp_model_for_uq(...) or get_model_ensemble_for_uq(...)
MODEL_NAME="HF"

# Concrete low-fidelity model choice, if the example module supports different options for the LF model
LF_MODEL_TYPE="dcopf"

# Solver used by SPAROW for all Sample Average Approximation (SAA) solves and xhat evaluations
SOLVER="ipopt"

# If true, print detailed progress/timing information to stdout/log
VERBOSE="true"

# -----------------------------------------------------------------------------
# Candidate-solution generation
# -----------------------------------------------------------------------------

# If false, the script generates a new candidate xhat by solving one SAA on a
# sampled scenario batch of size CANDIDATE_SCEN_COUNT using CANDIDATE_SEED.
# If true, the script loads an existing xhat from XHAT_FILE.
USE_EXISTING_XHAT="true"

# Number of sampled scenarios used to generate a new candidate xhat
CANDIDATE_SCEN_COUNT=1

# Seed used only for candidate-solution generation sampling
CANDIDATE_SEED=999

# Whether candidate-generation sampling should be with replacement or without
# replacement from the finite scenario population
CANDIDATE_WITH_REPLACEMENT="false"

# Path to candidate solution file (.npy dictionary). If USE_EXISTING_XHAT=false,
# this file will be created.
XHAT_FILE="${SCRIPT_DIR}/../../sparow_examples/sparow_examples/uq_opf/candidate_xhat.npy"

# -----------------------------------------------------------------------------
# Main experiment controls
# -----------------------------------------------------------------------------

# Significance level; confidence level is 1 - ALPHA
ALPHA=0.05

# Base seed used for the main budgeted experiment workflow
MRP_SEED=54321

# Whether replication batches used in the main workflow are sampled with
# replacement from the finite scenario population
MRP_WITH_REPLACEMENT="true"

# List of replication batch sizes n to test. Each replication-level estimator
# uses n iid sampled scenarios.
N_VALUES="2,3"

# Number of paired replications (HF and LF model evals on same sampled scenario batch)
# In single-fidelity experiments, this is just number of HF model replications)
M_VALUES="2,3"

# OPTIONAL: for multifidelity experiments, number of additional LF model replications
ACV_M_VALUES="2,3"

# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------

# These files should be in same folder as the script where your problem instance is defined
RESULTS_CSV="grid_results.csv" # this file gets automatically written to correct folder
CLI_LOG="${SCRIPT_DIR}/../../sparow_examples/sparow_examples/uq_opf/experiment_run.log"

# Filepaths for scripts containing plotting functionality
PLOT_SCRIPT_STANDARD="${SCRIPT_DIR}/plot_mrp_results.py"
PLOT_SCRIPT_ACV="${SCRIPT_DIR}/plot_acvmrp_results.py"

# =============================================================================
# Convert booleans into CLI flags
# =============================================================================

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
    python -m run_grid_experiments \
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
    python -m run_grid_experiments \
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

