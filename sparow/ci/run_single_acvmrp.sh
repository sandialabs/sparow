#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# User settings
# ==========================================================

MODEL_MODULE="sparow_examples.mrp_facilityloc.mrp_discrete_facilityloc"
MODEL_NAME="HF"
LF_MODEL_TYPE="classic"
SOLVER="gurobi_direct"

# Candidate-solution generation
CANDIDATE_SCEN_COUNT=1
CANDIDATE_SEED=123
CANDIDATE_WITH_REPLACEMENT="false"

# MRP confidence interval settings
ALPHA=0.05
MRP_SEED=678
MRP_WITH_REPLACEMENT="false"

# Single-run settings
SCENARIO_FILE="../../../sparow_examples/sparow_examples/mrp_facilityloc/discrete_facilityloc_scenarios.npy"
XHAT_FILE="manually_created_suboptimal_xhat.npy"
N=100
M_PAIRED=10
M_LF_ONLY=5
COMPUTE_TRUE_GAP="true"
ACV_MRP="true"

# ==========================================================
# Convert replacement choice into CLI flag
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

TRUE_GAP_FLAG=""
if [ "${COMPUTE_TRUE_GAP}" = "true" ]; then
    TRUE_GAP_FLAG="--compute-true-gap"
elif [ "${COMPUTE_TRUE_GAP}" = "false" ]; then
    TRUE_GAP_FLAG=""
else
    echo "Error: COMPUTE_TRUE_GAP must be 'true' or 'false'"
    exit 1nario_data.values
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

# ==========================================================
# Basic file existence checks
# ==========================================================

if [ ! -f "${SCENARIO_FILE}" ]; then
    echo "Error: scenario file not found: ${SCENARIO_FILE}"
    exit 1
fi

echo "=== Running single ACV-MRP experiment ==="
echo "ACV-MRP flag: ${ACV_FLAG}"
echo "Candidate sampling flag: ${CANDIDATE_FLAG}"
echo "MRP sampling flag: ${MRP_FLAG}"
echo "Scenario file: ${SCENARIO_FILE}"
echo "xhat file: ${XHAT_FILE}"
echo "candidate_scen_count: ${CANDIDATE_SCEN_COUNT}"
echo "n: ${N}"
echo "m: ${M_PAIRED}"
echo "M: ${M_LF_ONLY}"

python -m sparow.ci.cli \
    --model-module "${MODEL_MODULE}" \
    --model-name "${MODEL_NAME}" \
    --lf-model-type "${LF_MODEL_TYPE}" \
    --solver-name "${SOLVER}" \
    --candidate-scen-count "${CANDIDATE_SCEN_COUNT}" \
    --candidate-seed "${CANDIDATE_SEED}" \
    ${CANDIDATE_FLAG} \
    --alpha "${ALPHA}" \
    --mrp-seed "${MRP_SEED}" \
    ${MRP_FLAG} \
    --scenario-file "${SCENARIO_FILE}" \
    --xhat-file "${XHAT_FILE}" \
    --n "${N}" \
    --m "${M_PAIRED}" \
    --M "${M_LF_ONLY}" \
    ${TRUE_GAP_FLAG} \
    ${ACV_FLAG}

echo "Single ACV-MRP experiment complete."