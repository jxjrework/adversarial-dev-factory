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
if [ $# -ge 3 ] 
then
  INPUT_DIR=$1
  OUTPUT_DIR=$2
  MAX_EPSILON=$3
  NB_ITER=2
fi

if [ $# -eq 4 ] 
then
  NB_ITER=$4
fi

python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --nb_iter="${NB_ITER}" \
  --checkpoint_path1=inception_v3.ckpt \
  --checkpoint_path2=adv_inception_v3.ckpt \
  --checkpoint_path3=ens_adv_inception_resnet_v2.ckpt
