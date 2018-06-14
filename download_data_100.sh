#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Download checkpoints for sample attacks and defenses.
sample_attacks/download_checkpoints.sh
sample_targeted_attacks/download_checkpoints.sh
sample_defenses/download_checkpoints.sh

# Randomly download 100 images.
mkdir -p dataset/images
python ./dataset/download_images.py \
  --input_file=dataset/dev_dataset.csv \
  --numbers_images=100 \
  --output_dir=dataset/images/
