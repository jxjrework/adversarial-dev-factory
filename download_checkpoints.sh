#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Download checkpoints for sample attacks and defenses.
sample_attacks/download_checkpoints.sh
sample_defenses/download_checkpoints.sh
sample_targeted_attacks/download_checkpoints.sh

