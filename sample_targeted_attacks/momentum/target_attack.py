"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os, time

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf
from nets import inception_v3, inception_resnet_v2

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

FLAGS = tf.flags.FLAGS

def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


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


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def graph_large(x, target_class_input, i, x_max, x_min, grad):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  alpha = eps / 12
  momentum = FLAGS.momentum
  num_classes = 1001

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_v3, end_points_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        x, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
            
  one_hot_target_class = tf.one_hot(target_class_input, num_classes)

  logits = (4 * logits_v3 + logits_adv_v3 + logits_ens3_adv_v3 + logits_ens4_adv_v3 + 4 * logits_ensadv_res_v2) / 11
  auxlogits = (4 * end_points_v3['AuxLogits'] + end_points_adv_v3['AuxLogits'] + end_points_ens3_adv_v3['AuxLogits'] + end_points_ens4_adv_v3['AuxLogits'] + 4 * end_points_ensadv_res_v2['AuxLogits']) / 11
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                   auxlogits,
                                                   label_smoothing=0.0,
                                                   weights=0.4)
  noise = tf.gradients(cross_entropy, x)[0]
  noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), [FLAGS.batch_size, 1, 1, 1])
  noise = momentum * grad + noise
  noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), [FLAGS.batch_size, 1, 1, 1])
  x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, target_class_input, i, x_max, x_min, noise


def graph_small(x, target_class_input, i, x_max, x_min, grad):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  alpha = eps / 28
  momentum = FLAGS.momentum
  num_classes = 1001

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_v3, end_points_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        x, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
            
  one_hot_target_class = tf.one_hot(target_class_input, num_classes)

  logits = (logits_v3 + 2 * logits_ensadv_res_v2) / 3
  auxlogits = (end_points_v3['AuxLogits'] + 2 * end_points_ensadv_res_v2['AuxLogits']) / 3
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                   auxlogits,
                                                   label_smoothing=0.0,
                                                   weights=0.4)
  noise = tf.gradients(cross_entropy, x)[0]
  noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), [FLAGS.batch_size, 1, 1, 1])
  noise = momentum * grad + noise
  noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), [FLAGS.batch_size, 1, 1, 1])
  x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, target_class_input, i, x_max, x_min, noise


def stop_large(x, target_class_input, i, x_max, x_min, grad):
  return tf.less(i, 20)


def stop_small(x, target_class_input, i, x_max, x_min, grad):
  return tf.less(i, 40)


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  start_time = time.time()

  eps = 2.0 * FLAGS.max_epsilon / 255.0

  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  tf.logging.set_verbosity(tf.logging.INFO)

  all_images_taget_class = load_target_class(FLAGS.input_dir)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape)

    if FLAGS.max_epsilon >= 8:
      x_adv, _, _, _, _, _ = tf.while_loop(stop_large, graph_large, [x_input, target_class_input, i, x_max, x_min, grad])
      s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
      s2 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
      s3 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
      s4 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
      s5 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
    else:
      x_adv, _, _, _, _, _ = tf.while_loop(stop_small, graph_small, [x_input, target_class_input, i, x_max, x_min, grad])
      s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
      s2 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))

    # Run computation
    with tf.Session() as sess:
      if FLAGS.max_epsilon >= 8:
        s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
        s2.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
        s3.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
        s4.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
        s5.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
      else:
        s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
        s2.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)

      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        target_class_for_batch = (
            [all_images_taget_class[n] for n in filenames]
            + [0] * (FLAGS.batch_size - len(filenames)))
        adv_images = sess.run(x_adv, feed_dict={x_input: images, target_class_input: target_class_for_batch})
        save_images(adv_images, filenames, FLAGS.output_dir)

  elapsed_time = time.time() - start_time
  print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

if __name__ == '__main__':
  tf.app.run()
