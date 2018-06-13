#!/bin/bash -x

# exit on first error
set -e

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ATTACKS_DIR="${SCRIPT_DIR}/sample_attacks"
TARGETED_ATTACKS_DIR="${SCRIPT_DIR}/sample_targeted_attacks"
DEFENSES_DIR="${SCRIPT_DIR}/sample_defenses"
DATASET_DIR="${SCRIPT_DIR}/dataset/images"
DATASET_METADATA_FILE="${SCRIPT_DIR}/dataset/dev_dataset.csv"
MAX_EPSILON=32

# Prepare working directory and copy all necessary files.
# In particular copy attacks defenses and dataset, so originals won't
# be overwritten.
echo "Preparing working directory: ${SCRIPT_DIR}"
mkdir "${SCRIPT_DIR}/intermediate_results"
mkdir "${SCRIPT_DIR}/output_dir"

echo "Running attacks and defenses"
python "${SCRIPT_DIR}/run_attacks_and_defenses_local.py" \
  --attacks_dir="${ATTACKS_DIR}" \
  --targeted_attacks_dir="${TARGETED_ATTACKS_DIR}" \
  --defenses_dir="${DEFENSES_DIR}" \
  --dataset_dir="${DATASET_DIR}" \
  --intermediate_results_dir="${SCRIPT_DIR}/intermediate_results" \
  --dataset_metadata="${DATASET_METADATA_FILE}" \
  --output_dir="${SCRIPT_DIR}/output_dir" \
  --epsilon="${MAX_EPSILON}" \
  --save_all_classification

echo "Output is saved in directory '${SCRIPT_DIR}/output_dir'"
