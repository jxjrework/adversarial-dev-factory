#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

# For how many iterations run this attack
NUM_ITERATIONS=180

# two kinds of attck here, one using only InceptionResnet_v2, attack_random_padding.py
# another use Toshi method ensembling 3 classifiers: Inception v3, adv_inception_v3 and ens_adv_inception_resnet_v2
python attack_iter_target_class_working_stage_for_randomPadding_EOT_allModel_EnsembleSize_15_iteration_180.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --num_iter="${NUM_ITERATIONS}" \
  --checkpoint_path1=inception_v3.ckpt \
  --checkpoint_path2=adv_inception_v3.ckpt \
  --checkpoint_path3=ens_adv_inception_resnet_v2.ckpt
