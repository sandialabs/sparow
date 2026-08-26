#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Aggregate budgeted UQ CSV outputs
# ============================================================================
#
# This script is meant to be run AFTER all parallel budgeted-UQ jobs have finished.
#
# Each individual run of run_budgeted_uq_experiments.sh writes its own uniquely
# named CSV files, for example:
#   - budgeted_uq_summary__n_5__B_250.csv
#   - budgeted_uq_macrorep__n_5__B_250.csv
#
# This aggregation script:
#   1. finds all matching summary CSVs,
#   2. combines them into one summary CSV with a single header row,
#   3. finds all matching macro-replication CSVs,
#   4. combines them into one macro CSV with a single header row.
#
# It does not modify the original per-run files.
#
# Example usage:
#   bash aggregate_budgeted_uq_csvs.sh
#
# ============================================================================

# Root directory under which to search for CSVs
SEARCH_ROOT="${SEARCH_ROOT:-.}"

# Filename patterns for per-run outputs
SUMMARY_PATTERN="${SUMMARY_PATTERN:-budgeted_uq_summary__n_*__B_*.csv}"
MACRO_PATTERN="${MACRO_PATTERN:-budgeted_uq_macrorep__n_*__B_*.csv}"

# Names of the aggregated output files
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:-combined_budgeted_uq_summary.csv}"
OUTPUT_MACRO="${OUTPUT_MACRO:-combined_budgeted_uq_macrorep.csv}"

echo "=== Aggregating budgeted UQ CSV files ==="
echo "Search root: ${SEARCH_ROOT}"
echo "Summary pattern: ${SUMMARY_PATTERN}"
echo "Macro pattern: ${MACRO_PATTERN}"
echo "Output summary CSV: ${OUTPUT_SUMMARY}"
echo "Output macro CSV: ${OUTPUT_MACRO}"

# ----------------------------------------------------------------------------
# Aggregate summary CSVs
# ----------------------------------------------------------------------------

# Find all summary CSVs matching the requested pattern.
mapfile -t SUMMARY_FILES < <(find "${SEARCH_ROOT}" -name "${SUMMARY_PATTERN}" | sort)

if [ "${#SUMMARY_FILES[@]}" -eq 0 ]; then
    echo "No summary CSV files found."
else
    echo "Found ${#SUMMARY_FILES[@]} summary CSV file(s)."

    # Write the header from the first file once.
    head -n 1 "${SUMMARY_FILES[0]}" > "${OUTPUT_SUMMARY}"

    # Append data rows from every file, skipping each file's header.
    for f in "${SUMMARY_FILES[@]}"; do
        echo "Adding summary rows from: $f"
        tail -n +2 "$f" >> "${OUTPUT_SUMMARY}"
    done

    echo "Wrote aggregated summary CSV: ${OUTPUT_SUMMARY}"
fi

# ----------------------------------------------------------------------------
# Aggregate macro-replication CSVs
# ----------------------------------------------------------------------------

# Find all macro CSVs matching the requested pattern.
mapfile -t MACRO_FILES < <(find "${SEARCH_ROOT}" -name "${MACRO_PATTERN}" | sort)

if [ "${#MACRO_FILES[@]}" -eq 0 ]; then
    echo "No macro-replication CSV files found."
else
    echo "Found ${#MACRO_FILES[@]} macro-replication CSV file(s)."

    # Write the header from the first file once.
    head -n 1 "${MACRO_FILES[0]}" > "${OUTPUT_MACRO}"

    # Append data rows from every file, skipping each file's header.
    for f in "${MACRO_FILES[@]}"; do
        echo "Adding macro-replication rows from: $f"
        tail -n +2 "$f" >> "${OUTPUT_MACRO}"
    done

    echo "Wrote aggregated macro CSV: ${OUTPUT_MACRO}"
fi

echo "CSV aggregation complete."