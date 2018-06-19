#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download inception v3 checkpoint for step_target_class attack.
cd "${SCRIPT_DIR}/iter_target_class/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz

# Download checkpoints for momentum attack
# Original file is from Tshinghua http://ml.cs.tsinghua.edu.cn/~yinpeng/nips17/targeted/models.zip
# I convert it to tar.gz
cd "${SCRIPT_DIR}/momentum/"
python ../../google_drive_downloader.py 1TnT-nHf_a375ilDJKWp5jAu3Bjjwpe8y checkpoints.tar.gz
tar -xvzf checkpoints.tar.gz
rm checkpoints.tar.gz
