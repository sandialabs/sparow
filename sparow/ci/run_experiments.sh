#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# User settings
# ==========================================================

MODEL_MODULE="sparow_examples.farmers.MRPfarmers"
MODEL_NAME="Advanced"
SOLVER="highs"

# Candidate-solution generation
CANDIDATE_SCEN_COUNT=5
CANDIDATE_SEED=12345
WITH_REPLACEMENT="true"

# MRP confidence interval settings
ALPHA=0.05
MRP_SEED=54321

# Grid of m and n values
M_VALUES="10,15,20"
N_VALUES="100,200,300,400,500,600"

# Files
XHAT_FILE="candidate_xhat.npy"
RESULTS_CSV="mrp_grid_results.csv"
PLOT_SCRIPT="plot_mrp_results.py"

echo "=== Running experiments for standard MRP algorithm ==="

python -m sparow.ci.cli \
    --grid-experiment \
    --model-module "${MODEL_MODULE}" \
    --model-name "${MODEL_NAME}" \
    --solver-name "${SOLVER}" \
    --candidate-scen-count "${CANDIDATE_SCEN_COUNT}" \
    --candidate-seed "${CANDIDATE_SEED}" \
    --alpha "${ALPHA}" \
    --mrp-seed "${MRP_SEED}" \
    --with-replacement \
    --m-values "${M_VALUES}" \
    --n-values "${N_VALUES}" \
    --xhat-file "${XHAT_FILE}" \
    --output-csv "${RESULTS_CSV}"

echo "Wrote ${RESULTS_CSV}"
echo "Wrote ${XHAT_FILE}"

python "${PLOT_SCRIPT}" "${RESULTS_CSV}"
echo "Plots created."