
# Simplified development toolkit for participants of adversarial competition

This is modified from the development toolkit for the
[Competition on Adversarial Attacks and Defenses 2018](http://caad.geekpwn.org/)
Following software required to use this package:

This toolkit is based on toolkit for NIPS17 competition which was released as part of [Cleverhans](https://github.com/tensorflow/cleverhans/blob/master/LICENSE) library under [MIT License](https://github.com/tensorflow/cleverhans/blob/master/LICENSE)

* Dev dataset which participants can use for development and testing of their
  attacks and defenses
* Sample adversarial attacks
* Sample adversarial defenses
* Tool to run attacks against defenses and compute score.

## Installation

### Prerequisites

Following software required to use this package:

* Python 2.7 with installed [Numpy](http://www.numpy.org/)
  and [Pillow](https://python-pillow.org/) packages.
* [Docker](https://www.docker.com/)

Additionally, all provided examples are written with use of
the [TensorFlow](https://www.tensorflow.org/).
Thus you may find useful to install TensorFlow to experiment with the examples,
however this is not strictly necessary.

### Installation procedure

To be able to run the examples you need to download checkpoints for provided models
as well as dataset.

To download the dataset and all checkpoints run following:

```bash
./download_data.sh
```
To download randomly 100 images and all checkpoints run following:

```bash
./download_data_100.sh
```

If you only want to download checkpoints run following:

```bash
./download_ckpts.sh
```

If you only want to to download the dataset then you can run:

```bash
./download_images.sh <output_folder> <number of images>
```

## Dataset

This toolkit includes DEV dataset with 1000 labelled images.
DEV dataset could be used for development and testing of adversarial attacks
and defenses.

Details about dataset are [here](./dataset/README.md).

## Sample attacks and defenses

Toolkit includes examples of attacks and defenses in the following directories:

* `sample_attacks/` - directory with examples of attacks:
  * `sample_attacks/fgsm/` - Fast gradient sign attack.
  * `sample_attacks/noop/` - No-op attack, which just copied images unchanged.
  * `sample_attacks/MultiModelIterativeMethod/` - Submission for NIPS 2017: (5th place), which is based on BasicIterativeMethod included in CleverHans with following three modifications.
     [[code](https://github.com/toshi-k/kaggle-nips-2017-adversarial-attack)]
     * Three models are attacked; inception_v3.ckpt, adv_inception_v3.ckpt, ens_adv_inception_resnet_v2.ckpt.
     * Number of iteration is set 10 to finish attacking in time.
     * Gradient is smoothed spatially. This procedure make smoothed perturbation and encourage transferability.
* `sample_targeted_attacks/` - directory with examples of targeted attacks:
  * `sample_targeted_attacks/target_attack_EOT_toshi_on_randomPadding/` - Our implementation of EOT on Toshi attack. (Iteration=180 takes 1 min to generate 1 picture ) EOT: [[code](https://github.com/anishathalye/obfuscated-gradients)][[paper](https://arxiv.org/pdf/1802.00420.pdf)]
  * `sample_targeted_attacks/momentum/` - Submission for NIPS competition 2017 (1st place). 
     A novel momentum iterative method has been applied to avoid the local minimium.[[code](https://github.com/dongyp13/Targeted-Adversarial-Attack)][[paper](https://arxiv.org/pdf/1710.06081.pdf)]
  * `sample_targeted_attacks/target_class_toshi_k/` - Submission for NIPS 2017: (9th place), which is based on iter_target_class with following four modifications.
     [[code](https://github.com/toshi-k/kaggle-nips-2017-adversarial-attack)]
     * Three models are attacked; inception_v3.ckpt, adv_inception_v3.ckpt, ens_adv_inception_resnet_v2.ckpt.
     * Number of iteration is set 14 to finish attacking in time.
     * Gradient is smoothed spatially. This procedure make smoothed perturbation and encourage transferability.
     * Save method with Image (PIL) is used to save images, instead of imsave (scipy.misc).

  * `sample_targeted_attacks/iter_target_class/` - iterative target class
    attack. This is a pretty good white-box attack,
    but it does not do well in black box setting.
* `sample_defenses/` - directory with examples of defenses:
  * `sample_defenses/base_inception_model/` - baseline inception classifier,
    which actually does not provide any defense against adversarial examples.
  * `sample_defenses/adv_inception_v3/` - adversarially trained Inception v3
    model from Adversarial Machine Learning at Scale. [[paper](https://arxiv.org/abs/1611.01236)]
  * `sample_defenses/ens_adv_inception_resnet_v2/` - Inception ResNet v2
    model which is adversarially trained against an ensemble of different
    kind of adversarial examples. Model is described in
    Ensemble Adversarial Training: Attacks and Defenses. [[paper](https://arxiv.org/abs/1705.07204)]
  * `sample_defenses/Guided_Denoise/` - Submission for NIPS competition 2017 (1st place).
    A denoise network has been trained.[[code](https://github.com/lfz/Guided-Denoise)][[paper](https://arxiv.org/abs/1712.02976)]
  * `sample_defenses/Guided_Denoise_14/` - Submission for CAAD 2018. First place in the first round. Modified Guided_Denoise to keep model 1 and 4 only. Enhanced defense performance against Toshi targeted attack.
  * `sample_defenses/Random_padding_IresV2/` - Submission for NIPS competition 2017 (2nd place).
    The main ideal of the defense is to utilize randomization (random resizing and random padding) to defend adversarial examples.[[code](https://github.com/cihangxie/NIPS2017_adv_challenge_defense)][[paper](https://arxiv.org/pdf/1711.01991.pdf)]
  * `sample_defenses/Random_Guided_Denoise/` - Our implementaion of combination of randomization (random resizing and random padding) and Denoise. Iteration = 5 (10 seconds processing 16 pictures)

### Structure of attacks and defenses

Each attack and defense should be stored in a separate subdirectory,
should be self-contained and intended to be run inside Docker container.

Directory with each attack or defense should contain file `metadata.json`
in JSON format with following fields:

* `type` could be one of `"attack"`, `"defense"` or `"targeted_attack"`.
* `container` is a URL of Docker container inside which attack or defense
  should be run.
* `container_gpu` is an optional field, URL of Docker container with
  GPU support.
* `entry_point` is a script which launches attack or defense.

Example of `metadata.json`:

```json
{
  "type": "attack",
  "container": "gcr.io/tensorflow/tensorflow:1.1.0",
  "container_gpu": "gcr.io/tensorflow/tensorflow:1.1.0-gpu",
  "entry_point": "run_attack.sh"
}
```

#### Non-targeted attack

Entry point script for a non-targeted attack should accept three arguments:
input directory, output directory and maximum size of adversarial perturbation
(in [infinity norm](https://en.wikipedia.org/wiki/Uniform_norm)). It will be
invoked in the following way:

```bash
attack_entry_point.sh INPUT_DIR OUTPUT_DIR MAX_SIZE_OF_PERTURBAION
```

Input directory will contain source images from dataset in PNG format and attack
has to write adversarial images into output directory.
Input images are 299x299 pixels RGB images, output images should have the same
size and also written in PNG format.
Filenames of adversarial images should be the same as filenames of
corresponding source images from the dataset.

Non-targeted attack is expected to produce adversarial images which are likely
will be misclassified by image classifier (assuming that it can classify source
images well).

Difference between each generated adversarial images and corresponding source
image has to be within specified maximum size of adversarial perturbation.
If it's not the case then competition runtime will automatically clip
adversarial image to be within the limits.

#### Targeted attack

Entry point script for a targeted attack accepts the same set of arguments as
for non-targeted attack: input directory, output directory, maximum size of
perturbation.

The only difference is that input directory will contain `target_class.csv` file
addition to images. Each line of `target_class.csv` will contain
comma-separated pairs of image filename and target class.

Targeted attack is expected to produce adversarial image which will
be likely classified as desired target class by image classifier.

Difference between source images and generated adversarial images
should be within specified maximum size of perturbation,
similarly to non-targeted attack.

#### Defense

Entry point script for a defense accepts two arguments: input directory and
output file. It will be invoked in a following way:

```bash
defense_entry_point.sh INPUT_DIR OUTPUT_FILE
```

Input directory will contain bunch of adversarial images in PNG format.
Defense has to classify all these images and write its predictions into
output file. Each line of the output file should contain comma separated image
filename and predicted label.

## How to run attacks against defenses

Script `run_attacks_and_defenses.py` runs all attacks against all defenses **in Docker**
and computes scores of each attack and each defense. You can run `run_attacks_and_defenses_local.py` to run them locally.

You can also run it in a following way:

```bash
python run_attacks_and_defenses.py \
  --attacks_dir="${DIRECTORY_WITH_ATTACKS}" \
  --targeted_attacks_dir="${DIRECTORY_WITH_TARGETED_ATTACKS}" \
  --defenses_dir="${DIRECTORY_WITH_DEFENSES}" \
  --dataset_dir="${DIRECTORY_WITH_DATASET_IMAGES}" \
  --intermediate_results_dir="${TEMP_DIRECTORY_FOR_INTERMEDIATE_RESULTS}" \
  --dataset_metadata=dataset/dataset.csv \
  --output_dir="${OUTPUT_DIRECTORY}" \
  --epsilon="${MAXIMUM_SIZE_OF_ADVERSARIAL_PERTURBATION}"
```

If you have GPU card and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed then you can
additionally pass `--gpu` argument to `run_attacks_and_defenses.py`
so attacks and defenses will be able to take advantage of GPU to speedup
computations.

Alternatively instead of running `run_attacks_and_defenses.py` directly and
providing all command line arguments you can use helper script
`run_attacks_and_defenses.sh` to run all attacks and defenses from this toolkit
against each other and save results to temporary directory.

NOTE: You should cleanup temporary directory created by
`run_attacks_and_defenses.sh` after running it.

`run_attacks_and_defenses.py` will write following files into output directory:

* `accuracy_on_attacks.csv` with matrix which will contain number of correctly
  classified images for each pair of non-targeted attack and defense.
  Columns of the matrix are defenses, rows of the matrix are
  non-targeted attacks.
* `accuracy_on_targeted_attacks.csv` with matrix which will contain number of
  correctly classified images for each pair of targeted attack and defense.
  Columns of the matrix are defenses, rows of the matrix are targeted attacks.
* `hit_target_class.csv` with matrix which will contain number of times images
  were classified as target class by defense for each given targeted attack.
  Columns of the matrix are defenses, rows of the matrix are targeted attacks.
* `defense_ranking.csv` with ranking of all defenses (best - first,
  worst - last, ties in arbitrary order), along with the score of each defense.
  Score for each defense is computed as total number of correctly classified
  adversarial images by defense classifier.
* `attack_ranking.csv` with ranking of all non-targeted (best - first,
  worst - last, ties in arbitrary order), along with the score of each attack.
  Score for each attack is computed as total number of time attack was able to
  cause incorrect classification
* `targeted_attack_ranking.csv` with ranking of all targeted attacks
  (best - first, worst - last, ties in arbitrary order), along with the score of
  each targeted attack.
  Score is computed as number of times the attack was able to force defense
  classifier to recognize adversarial image as specified target class.

Additionally, if flag `--save_all_classification` is provided then
`run_attacks_and_defenses.py` will save file `all_classification.csv`
which contains classification predictions (along with true classes and
target classes) for each adversarial image generated by each attack
and classified by each defense. This might be useful for debugging.
