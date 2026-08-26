#!/usr/bin/env bash
set -euo pipefail

# For a single parameter configuration, run multifidelity ACV-MRP
# Results are displayed in the terminal, not written to file, for quick checks

# ==========================================================
# User settings
# ==========================================================

# Python module that defines: get_model_ensemble_for_uq(...)
MODEL_MODULE="sparow_examples.uq_facilityloc.uq_discrete_facilityloc"

# Optional model name passed into get_model_ensemble_for_uq(...)
MODEL_NAME="HF"

# Concrete low-fidelity model choice, if the example module supports different options for the LF model
LF_MODEL_TYPE=""

# Solver used by SPAROW for all Sample Average Approximation (SAA) solves and xhat evaluations
SOLVER="gurobi_direct"

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
XHAT_FILE="../../sparow_examples/sparow_examples/uq_facilityloc/manually_created_suboptimal_xhat.npy"

# -----------------------------------------------------------------------------
# Main experiment controls
# -----------------------------------------------------------------------------

# Significance level; confidence level is 1 - ALPHA
ALPHA=0.05

# Base seed used for the main budgeted experiment workflow
MRP_SEED=678

# Whether replication batches used in the main workflow are sampled with
# replacement from the finite scenario population
MRP_WITH_REPLACEMENT="true"

# ------------------------------------------------------------------------------------
# Single-run Settings (i.e. - single parameter configuration, not a parameter sweep)
# ------------------------------------------------------------------------------------

# Each replication-level estimator uses n iid sampled scenarios.
N=100 

# Number of paired replications (HF and LF model evals on same sampled scenario batch)
M_PAIRED=30 

# Number of additional LF model replications
M_LF_ONLY=10 

# If true, compute the exact finite-population true gap once for benchmarking
COMPUTE_TRUE_GAP="true" 

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

TRUE_GAP_FLAG=""
if [ "${COMPUTE_TRUE_GAP}" = "true" ]; then
    TRUE_GAP_FLAG="--compute-true-gap"
elif [ "${COMPUTE_TRUE_GAP}" = "false" ]; then
    TRUE_GAP_FLAG=""
else
    echo "Error: COMPUTE_TRUE_GAP must be 'true' or 'false'"
    exit 1
fi

echo "=== Running single ACV-MRP experiment ==="
echo "Candidate sampling flag: ${CANDIDATE_FLAG}"
echo "MRP sampling flag: ${MRP_FLAG}"
echo "xhat file: ${XHAT_FILE}"
echo "candidate_scen_count: ${CANDIDATE_SCEN_COUNT}"
echo "n: ${N}"
echo "m: ${M_PAIRED}"
echo "M: ${M_LF_ONLY}"

python -m run_grid_experiments \
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
    --m "${M_PAIRED}" \
    --M "${M_LF_ONLY}" \
    ${TRUE_GAP_FLAG} \
    "--acv-mrp"

echo "Single ACV-MRP experiment complete."