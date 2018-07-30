#!/bin/sh
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python-pip
sudo pip install --upgrade pip

# fundamental tools
sudo apt-get install -y python-numpy python-scipy python-nose
#sudo apt-get install -y python-matplotlib ipython ipython-notebook python-pandas python-sympy
sudo pip install pillow
sudo pip install -U tensorflow

# gpu installation
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  sudo apt-get update
  sudo apt-get install -y cuda-9-0
fi
# Enable persistence mode
nvidia-smi -pm 1

# install docker CE
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce

# install nvidia docker
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
# Test nvidia-smi with the latest official CUDA image
sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

# package from github
echo "start downloading from github"
sudo git clone https://4530e8b0d1e63be298f40a7e142b5119542c4616@github.com/selfstudyjiao/adversarial-dev-factory.git
# cd adversarial-dev-factory
# sudo ./download_data_100.sh
# cd ..

# revise the package for testing
# attack
# sudo rm -r -f adversarial-dev-factory/sample_attacks/fgsm
sudo rm -r -f adversarial-dev-factory/sample_attacks/noop
sudo rm -r -f adversarial-dev-factory/sample_attacks/MultiModelIterativeMethod

# targeted attack
sudo rm -r -f adversarial-dev-factory/sample_targeted_attacks/iter_target_class
sudo rm -r -f adversarial-dev-factory/sample_targeted_attacks/jing_targeted
# sudo rm -r -f adversarial-dev-factory/sample_targeted_attacks/momentum
sudo rm -r -f adversarial-dev-factory/sample_targeted_attacks/target_attack_EOT_toshi_on_randomPadding
sudo rm -r -f adversarial-dev-factory/sample_targeted_attacks/target_attack_guided_denoise
sudo rm -r -f adversarial-dev-factory/sample_targeted_attacks/target_attack_jpeg_toshi
sudo rm -r -f adversarial-dev-factory/sample_targeted_attacks/target_class_toshi_k
sudo rm -r -f adversarial-dev-factory/sample_targeted_attacks/target_class_toshi_k_Sangxia

# defense
# sudo rm -r -f adversarial-dev-factory/sample_defenses/base_inception_model
sudo rm -r -f adversarial-dev-factory/sample_defenses/defense_bitDepthReduction
sudo rm -r -f adversarial-dev-factory/sample_defenses/defense_crop
sudo rm -r -f adversarial-dev-factory/sample_defenses/defense_JPEG
sudo rm -r -f adversarial-dev-factory/sample_defenses/Diff_Random_Denoise_14
sudo rm -r -f adversarial-dev-factory/sample_defenses/ens_adv_inception_resnet_v2
sudo rm -r -f adversarial-dev-factory/sample_defenses/Guided_Denoise
sudo rm -r -f adversarial-dev-factory/sample_defenses/Guided_Denoise_14
sudo rm -r -f adversarial-dev-factory/sample_defenses/Random_Guided_Denoise
sudo rm -r -f adversarial-dev-factory/sample_defenses/Random_padding_IresV2
sudo rm -r -f adversarial-dev-factory/sample_defenses/tv

# start running attack & defense with gpu
sudo ./run_attacks_and_defenses_gpu.sh
