#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# User settings
# ==========================================================

MODEL_MODULE="sparow_examples.farmers.MRPfarmers"
MODEL_NAME="Advanced"
SOLVER="gurobi_direct"

# MRP confidence interval settings
ALPHA=0.05
MRP_SEED=678
MRP_WITH_REPLACEMENT="true"

# Single-run settings
SCENARIO_FILE="../../../sparow_examples/sparow_examples/farmers/advanced_farmers_1000_scenarios.npy"
XHAT_FILE="candidate_xhat.npy"
N=1000
M=20
COMPUTE_TRUE_GAP="true"

# ==========================================================
# Convert replacement choice into CLI flag
# ==========================================================

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
echo "MRP sampling flag: ${MRP_FLAG}"
echo "Scenario file: ${SCENARIO_FILE}"
echo "xhat file: ${XHAT_FILE}"
echo "n: ${N}"
echo "m: ${M}"

python -m sparow.ci.cli \
    --model-module "${MODEL_MODULE}" \
    --model-name "${MODEL_NAME}" \
    --solver-name "${SOLVER}" \
    --alpha "${ALPHA}" \
    --mrp-seed "${MRP_SEED}" \
    ${MRP_FLAG} \
    --scenario-file "${SCENARIO_FILE}" \
    --xhat-file "${XHAT_FILE}" \
    --n "${N}" \
    --m "${M}" \
    ${TRUE_GAP_FLAG}

echo "Single MRP experiment complete."