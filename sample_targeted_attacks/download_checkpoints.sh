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

# Download checkpoints for toshi_k target_class attack.
cd "${SCRIPT_DIR}/target_class_toshi_k/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz


# Download checkpoints for target_attack_EOT_toshi_on_randomPadding
cd "${SCRIPT_DIR}/target_attack_EOT_toshi_on_randomPadding/"
cp ../target_class_toshi_k/*.tar.gz .
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for target_attack_EOT_toshi_on_jpeg
cd "${SCRIPT_DIR}/target_attack_jpeg_toshi/"
mv ../target_class_toshi_k/*.tar.gz .
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download and make ensembled checkpoints for jing_target attack.
cd "${SCRIPT_DIR}/jing_targeted/"
# Download inception v3 checkpoint, Top-1 Accuracy: 78.0
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz -q
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
# Download adversarially trained inception v3 checkpoint
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz -q
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
# Download ensemble adversarially trained inception resnet v2 checkpoint
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz -q
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

echo "Creating ensemble model checkpoints"
# create ensemble model
python ensemble_models_v0.py
echo "ensemble model checkpoints established"
rm adv_inception_v3.ckpt.data-00000-of-00001
rm adv_inception_v3.ckpt.index
rm adv_inception_v3.ckpt.meta
rm ens_adv_inception_resnet_v2.ckpt.data-00000-of-00001
rm ens_adv_inception_resnet_v2.ckpt.index
rm ens_adv_inception_resnet_v2.ckpt.meta
rm inception_v3.ckpt

# Download checkpoints for target_attack_guided_denoise
cd "${SCRIPT_DIR}/target_attack_guided_denoise/"
mv ../../sample_defenses/Random_Guided_Denoise/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz
rm checkpoints.tar.gz
