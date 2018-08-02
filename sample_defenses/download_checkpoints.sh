#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download inception v3 checkpoint into base_inception_model subdirectory
cd "${SCRIPT_DIR}/base_inception_model/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz

# Also copy to defense tv
cd "${SCRIPT_DIR}/tv/"
mv ../base_inception_model/inception_v3_2016_08_28.tar.gz .
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz

# Download adversarially trained inception v3 checkpoint
# into adv_inception_v3 subdirectory
cd "${SCRIPT_DIR}/adv_inception_v3/"
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz

# Download ensemble adversarially trained inception resnet v2 checkpoint
# into ens_adv_inception_resnet_v2 subdirectory
cd "${SCRIPT_DIR}/ens_adv_inception_resnet_v2/"
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz


# Also copy to defense random_padding_iresV2
cd "${SCRIPT_DIR}/Random_padding_IresV2/"
cp ../ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2_2017_08_18.tar.gz .
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for Guided_denoise defense
# Original file is from https://www.dropbox.com/sh/q9ssnbhpx8l515t/AACvjiMmGRCteaApmj1zTrLTa?dl=0
# or https://pan.baidu.com/s/1hs7ti5Y#list/path=%2F
# I convert it to tar.gz
cd "${SCRIPT_DIR}/Guided_Denoise/"
python ../../google_drive_downloader.py 1p1zhtUeBA8MJa0p3X2WHxSoanIsEjH38 checkpoints.tar.gz
tar -xvzf checkpoints.tar.gz

# Same checkpoint for Guided Denoise with 14 models only
cd "${SCRIPT_DIR}/Guided_Denoise_14/"
mv ../Guided_Denoise/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz

# Same checkpoint for Random_Guided_Denoise
cd "${SCRIPT_DIR}/Random_Guided_Denoise/"
mv ../Guided_Denoise_14/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz

# copy ckpts for jpeg
cd "${SCRIPT_DIR}/defense_JPEG/"
cp ../ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2_2017_08_18.tar.gz .
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# copy ckpts for crop
cd "${SCRIPT_DIR}/defense_crop/"
cp ../ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2_2017_08_18.tar.gz .
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# copy ckpts for bit depth
cd "${SCRIPT_DIR}/defense_bitDepthReduction/"
cp ../ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2_2017_08_18.tar.gz .
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# checkpoints for Diff_Random_Denoise_14
cd "${SCRIPT_DIR}/Diff_Random_Denoise_14/"
mv ../Random_Guided_Denoise/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz
mv ../ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2_2017_08_18.tar.gz .
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# checkpoints for Diff_cv2_Random_Denoise_14_pytorch
cd "${SCRIPT_DIR}/Diff_Random_Denoise_14_pytorch/"
mv ../Diff_cv2_Random_Denoise_14/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz

# checkpoints for Diff_Random_Denoise_14_pytorch
cd "${SCRIPT_DIR}/Diff_Random_Denoise_14_pytorch/"
mv ../Diff_Random_Denoise_14/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz
rm checkpoints.tar.gz
