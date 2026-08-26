#!/usr/bin/env bash
set -euo pipefail

# This script runs budget-aware numerical experiments (parameter sweeps).
# This script automatically writes results to CSV files, but it does NOT automatically generate plots.
# To generate plots, get the resulting summary CSVs from this script, and then run with sparow/bin/plot_budgeted_uq_results.py.

# NOTE: you can specify the number of macro-replications, which repeats each fixed 
# parameter configuration across multiple seeds.
# This matters because a 95% confidence level is a repeated-sampling statement:
# over many repeated runs, about 95 out of 100 intervals are expected to contain
# the true population quantity, so macro-replications are useful for checking this behavior in practice.

# =============================================================================
# User settings
# =============================================================================

# Python module that defines: get_model_ensemble_for_uq(...)
MODEL_MODULE="sparow_examples.uq_opf.uq_opf"

# Optional model name passed into get_model_ensemble_for_uq(...)
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
CANDIDATE_SEED=123

# Whether candidate-generation sampling should be with replacement or without
# replacement from the finite scenario population
CANDIDATE_WITH_REPLACEMENT="false"

# Path to candidate solution file (.npy dictionary). If USE_EXISTING_XHAT=false,
# this file will be created.
XHAT_FILE="../../sparow_examples/sparow_examples/uq_opf/candidate_xhat.npy"

# -----------------------------------------------------------------------------
# Main experiment controls
# -----------------------------------------------------------------------------

# Significance level; confidence level is 1 - ALPHA
ALPHA=0.05

# Base seed used for the main budgeted experiment workflow
MAIN_SEED=678

# Whether replication batches used in the main workflow are sampled with
# replacement from the finite scenario population
MRP_WITH_REPLACEMENT="true"

# Number of macro-replications used to estimate empirical performance metrics
# such as coverage, average upper bound, and realized improvement probability
MACRO_REPLICATIONS=3

# List of replication batch sizes n to test. Each replication-level estimator
# uses n iid sampled scenarios.
N_VALUES="5"

# List of total wall-clock budgets to test. PyApprox uses these budgets to
# recommend multifidelity allocations (m, M).
BUDGET_VALUES="200,220"

# Number of shared pilot samples used to estimate covariance/correlation and
# model costs before allocating the budget
N_PILOT=10

# If true, the total budget includes the pilot-study cost.
# If false, the pilot study is treated as outside the reported budget.
COUNT_PILOT_COST_AGAINST_BUDGET="false"

# If true, re-use one pilot study across all macro-replications for each fixed n.
# If false, re-run a fresh pilot study inside each macro-replication.
REUSE_PILOT_ACROSS_MACROREPS="true"

# If true, compute the exact finite-population true gap once for benchmarking
COMPUTE_TRUE_GAP="true"

# If true, write a JSON file with extra diagnostic metadata and timing info
SAVE_DEBUG_JSON="true"

# -----------------------------------------------------------------------------
# Optional artificial delays for PyApprox cost-estimation experiments
# -----------------------------------------------------------------------------
# These are useful when debugging budget allocation behavior in examples where
# HF and LF solve times are too similar to clearly separate.
HF_COST_DELAY_SECONDS=0.0
LF_COST_DELAY_SECONDS=0.0

# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------

# Summary CSV aggregated over macro-replications
OUTPUT_SUMMARY_CSV="budgeted_uq_summary.csv"

# Per-macro-replication CSV
OUTPUT_MACRO_CSV="budgeted_uq_macrorep.csv"

# Optional debug JSON
DEBUG_JSON_FILE="budgeted_uq_debug.json"

# =============================================================================
# Convert booleans into CLI flags
# =============================================================================

VERBOSE_FLAG=""
if [ "${VERBOSE}" = "true" ]; then
    VERBOSE_FLAG="--verbose"
elif [ "${VERBOSE}" != "false" ]; then
    echo "Error: VERBOSE must be 'true' or 'false'"
    exit 1
fi

USE_EXISTING_XHAT_FLAG=""
if [ "${USE_EXISTING_XHAT}" = "true" ]; then
    USE_EXISTING_XHAT_FLAG="--use-existing-xhat"
elif [ "${USE_EXISTING_XHAT}" != "false" ]; then
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

COUNT_PILOT_FLAG=""
if [ "${COUNT_PILOT_COST_AGAINST_BUDGET}" = "true" ]; then
    COUNT_PILOT_FLAG="--count-pilot-cost-against-budget"
elif [ "${COUNT_PILOT_COST_AGAINST_BUDGET}" != "false" ]; then
    echo "Error: COUNT_PILOT_COST_AGAINST_BUDGET must be 'true' or 'false'"
    exit 1
fi

REUSE_PILOT_FLAG=""
if [ "${REUSE_PILOT_ACROSS_MACROREPS}" = "true" ]; then
    REUSE_PILOT_FLAG="--reuse-pilot-across-macroreps"
elif [ "${REUSE_PILOT_ACROSS_MACROREPS}" = "false" ]; then
    REUSE_PILOT_FLAG="--redo-pilot-per-macrorep"
else
    echo "Error: REUSE_PILOT_ACROSS_MACROREPS must be 'true' or 'false'"
    exit 1
fi

TRUE_GAP_FLAG=""
if [ "${COMPUTE_TRUE_GAP}" = "true" ]; then
    TRUE_GAP_FLAG="--compute-true-gap"
elif [ "${COMPUTE_TRUE_GAP}" != "false" ]; then
    echo "Error: COMPUTE_TRUE_GAP must be 'true' or 'false'"
    exit 1
fi

SAVE_DEBUG_FLAG=""
if [ "${SAVE_DEBUG_JSON}" = "true" ]; then
    SAVE_DEBUG_FLAG="--save-debug-json"
elif [ "${SAVE_DEBUG_JSON}" != "false" ]; then
    echo "Error: SAVE_DEBUG_JSON must be 'true' or 'false'"
    exit 1
fi

echo "=== Running budgeted UQ experiments ==="
echo "Model module: ${MODEL_MODULE}"
echo "Model name: ${MODEL_NAME}"
echo "LF model type: ${LF_MODEL_TYPE}"
echo "Solver: ${SOLVER}"
echo "Use existing xhat: ${USE_EXISTING_XHAT}"
echo "xhat file: ${XHAT_FILE}"
echo "Candidate scenario count: ${CANDIDATE_SCEN_COUNT}"
echo "Candidate seed: ${CANDIDATE_SEED}"
echo "Main seed: ${MAIN_SEED}"
echo "Macro replications: ${MACRO_REPLICATIONS}"
echo "n values: ${N_VALUES}"
echo "Budget values: ${BUDGET_VALUES}"
echo "Pilot samples: ${N_PILOT}"

python -m run_budgeted_uq_experiments \
    --model-module "${MODEL_MODULE}" \
    --model-name "${MODEL_NAME}" \
    --lf-model-type "${LF_MODEL_TYPE}" \
    --solver-name "${SOLVER}" \
    ${VERBOSE_FLAG} \
    --xhat-file "${XHAT_FILE}" \
    ${USE_EXISTING_XHAT_FLAG} \
    --candidate-scen-count "${CANDIDATE_SCEN_COUNT}" \
    --candidate-seed "${CANDIDATE_SEED}" \
    ${CANDIDATE_FLAG} \
    --alpha "${ALPHA}" \
    --main-seed "${MAIN_SEED}" \
    ${MRP_FLAG} \
    --macro-replications "${MACRO_REPLICATIONS}" \
    --n-values "${N_VALUES}" \
    --budget-values "${BUDGET_VALUES}" \
    --n-pilot "${N_PILOT}" \
    ${COUNT_PILOT_FLAG} \
    ${REUSE_PILOT_FLAG} \
    ${TRUE_GAP_FLAG} \
    ${SAVE_DEBUG_FLAG} \
    --hf-cost-delay-seconds "${HF_COST_DELAY_SECONDS}" \
    --lf-cost-delay-seconds "${LF_COST_DELAY_SECONDS}" \
    --output-summary-csv "${OUTPUT_SUMMARY_CSV}" \
    --output-macro-csv "${OUTPUT_MACRO_CSV}" \
    --debug-json-file "${DEBUG_JSON_FILE}"

echo "Budgeted UQ experiment run complete."