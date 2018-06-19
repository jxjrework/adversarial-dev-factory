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

# Another copy of inception v3 checkpoint for iter_target_class attack
mv inception_v3_2016_08_28.tar.gz "${SCRIPT_DIR}/momentum/"
cd "${SCRIPT_DIR}/momentum/"
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
# adv_inception_v3
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
# ens-adv_inception_resnet_v2
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
# ens3_adv_inception_v3
wget http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens3_adv_inception_v3_2017_08_18.tar.gz
rm ens3_adv_inception_v3_2017_08_18.tar.gz
# ens4_adv_inception_v3
wget http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens4_adv_inception_v3_2017_08_18.tar.gz
rm ens4_adv_inception_v3_2017_08_18.tar.gz

