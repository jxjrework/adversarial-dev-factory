"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread

import tensorflow as tf

import inception_resnet_v2

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

tf.flags.DEFINE_string(
    'filter', '', 'Method used to preprocessing the image.')

FLAGS = tf.flags.FLAGS

# impage processing with ski-image
def denoiser(denoiser_name, img, sigma):
    # For bilateral: sigma is sigma_spatial : float 
    #     Standard deviation for range distance. 
    #     A larger value results in averaging of pixels with larger spatial differences.
    # For wavelet: sigma is The noise standard deviation used 
    #     when computing the wavelet detail coefficient threshold(s)
    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, wiener, estimate_sigma)
    batch_shape = img.shape
    images = np.zeros(batch_shape)
    batch_size = batch_shape[0]
    for idx in range(batch_size):
      # change to [0 1] for image processing
      img[idx, :, :, :] = (img[idx, :, :, :] + 1 ) / 2.0
      sigma1 = estimate_sigma(img[idx, :, :, :], average_sigmas=False, multichannel=True )
      if denoiser_name == 'wavelet':
        images[idx, :, :, :] = denoise_wavelet(img[idx,:, :, :],sigma=sigma1, mode='soft', multichannel=True,convert2ycbcr=True, method='BayesShrink')
      elif denoiser_name == 'TVM':
        images[idx, :, :, :] = denoise_tv_chambolle(img[idx,:, :, :], multichannel=True)
      elif denoiser_name == 'bilateral':
        images[idx, :, :, :] = denoise_bilateral(img[idx,:, :, :], sigma_spatial=sigma, bins=1000, multichannel=True)
      elif denoiser_name == 'deconv':
        images[idx, :, :, :] = wiener(img[idx,:, :, :])
      elif denoiser_name == 'NLM':
        images[idx, :, :, :] = denoise_nl_means(img[idx,:, :, :], multichannel=True)
      else:
        raise Exception('Incorrect denoiser mentioned. Options: wavelet, TVM, bilateral, deconv, NLM')
      # change back to [-1 1]
      images[idx, :, :, :] = images[idx, :, :, :] * 2.0 - 1.0   
    return images

# image processing with cv2
def denoiser_cv2(denoiser_name, img):
    # For bilateral: sigma is sigma_spatial : float 
    #     Standard deviation for range distance. 
    #     A larger value results in averaging of pixels with larger spatial differences.
    # For wavelet: sigma is The noise standard deviation used 
    #     when computing the wavelet detail coefficient threshold(s)
    from cv2 import (bilateralFilter, fastNlMeansDenoisingColored, split, merge)
    batch_shape = img.shape
    images = np.zeros(batch_shape)
    batch_size = batch_shape[0]
    for idx in range(batch_size):
      # change to [0 1] for image processing
      img[idx, :, :, :] = (img[idx, :, :, :] + 1 ) / 2.0
      np_img = np.reshape(img[idx, :, :, :], (299, 299, 3)).astype(np.float32)
      if denoiser_name == 'bilateral':
        images[idx, :, :, :] = np.reshape(bilateralFilter(np_img, 9, 75, 75), (1, 299, 299, 3))
      elif denoiser_name == 'fastNLM':
        b,g,r = split(np_img)           # get b,g,r
        rgb_img = merge([r,g,b])     # switch it to rgb
        dst = fastNlMeansDenoisingColored(rgb_img,None,10,10,7,21)
        b,g,r = split(dst)           # get b,g,r
        rgb_dst = merge([r,g,b])     # switch it back
        images[idx, :, :, :] = np.reshape(rgb_dst, (1, 299, 299, 3))
      else:
        raise Exception('Incorrect denoiser mentioned. Options: bilateral,fastNlM')
      # change back to [-1 1]
      images[idx, :, :, :] = images[idx, :, :, :] * 2.0 - 1.0   
    return images

def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      _, end_points = inception_resnet_v2.inception_resnet_v2(
          x_input, num_classes=num_classes, is_training=False)

    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):

          # denoiser
          denoiser_cv2(FLAGS.filter, images)
          
          labels = sess.run(predicted_labels, feed_dict={x_input: images})
          for filename, label in zip(filenames, labels):
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
  tf.app.run()
