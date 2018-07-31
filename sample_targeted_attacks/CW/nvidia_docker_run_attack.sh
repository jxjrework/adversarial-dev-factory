#!/bin/bash -x
#
# docker_run_attack.sh is a script which executes the attack in nvidia-docker
#
# Envoronment which runs attacks and defences calls it in a following way:
#   docker_run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

nvidia-docker run -v ${INPUT_DIR}:/input_images \
                  -v ${OUTPUT_DIR}:/output_images \
                  -v `cd "$(dirname "$0")" && pwd`:/code \
                  -w /code \
                  xjiao/cleverhans:gpu \
                  ./run_attack.sh \
                  /input_images \
                  /output_images ${MAX_EPSILON}