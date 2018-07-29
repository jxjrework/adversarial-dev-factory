#!/bin/bash

# exit on first error
set -e

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAX_EPSILON=32
# This has to be full path/absolute path
ROOT_DIR=$1

echo "Running attacks and defenses"
python "${SCRIPT_DIR}/run_new_attacks_and_defenses_local.py" \
  --root_dir="${ROOT_DIR}" \
  --epsilon="${MAX_EPSILON}" \
  --save_all_classification

echo "Output is saved in directory '${ROOT_DIR}/output_dir'"
