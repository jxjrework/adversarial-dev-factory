"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils import set_log_level
import numpy as np
from PIL import Image

import logging

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

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
    'batch_size', 1, 'How many images process at one time.')

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
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      logits, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    #output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    #probs = output.op.inputs[0]
    return logits

class IrNetModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, scope=''):
        self.num_classes = num_classes
        self.built = False
        self.scope = scope

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=self.num_classes, reuse=reuse, is_training=False, scope=self.scope)

        self.built = True
        return logits

def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  all_images_taget_class = load_target_class(FLAGS.input_dir)

  with tf.Graph().as_default():
    # # Prepare graph
    # x_input = tf.placeholder(tf.float32, shape=batch_shape)

    # model = InceptionModel(num_classes)

    # x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    # x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    # cw = CarliniWagnerL2(model)
    
    # target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    # one_hot_target_class = tf.one_hot(target_class_input, num_classes)
    # cw_params = {'binary_search_steps': 1,
    #              'y_target': one_hot_target_class,
    #              'max_iterations': 10,
    #              'learning_rate': 0.1,
    #              'batch_size': FLAGS.batch_size,
    #              'initial_const': 10,
    #              'clip_min': -1.,
    #              'clip_max': 1.}
    # x_adv = cw.generate(x_input, **cw_params)
    
    # # Run computation
    # init_op = tf.global_variables_initializer()
    # saver = tf.train.Saver(slim.get_model_variables())
    # session_creator = tf.train.ChiefSessionCreator(
    #     scaffold=tf.train.Scaffold(saver=saver, local_init_op=init_op),
    #     checkpoint_filename_with_path=FLAGS.checkpoint_path,
    #     master=FLAGS.master)

    # with tf.train.MonitoredSession(session_creator=session_creator) as sess:
    #   for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    #     target_class_for_batch = (
    #                   [all_images_taget_class[n] for n in filenames] + [0] * (FLAGS.batch_size - len(filenames)))
    #     adv_images = sess.run(x_adv, feed_dict={x_input: images, target_class_input: target_class_for_batch})
    #     save_images(adv_images, filenames, FLAGS.output_dir)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")
    set_log_level(logging.DEBUG)

    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    model = InceptionModel(num_classes)
    #model =  IrNetModel(num_classes, scope='sc1')
 
    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, back='tf', sess=sess)
    
    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    one_hot_target_class = tf.one_hot(target_class_input, num_classes)
    cw_params = {'binary_search_steps': 3,
                 'y_target': one_hot_target_class,
                 'max_iterations': 200,
                 'learning_rate': 0.001,
                 'batch_size': FLAGS.batch_size,
                 'initial_const': 0.05,
                 'clip_min': -1.,
                 'clip_max': 1.}
    x_adv = cw.generate(x_input, **cw_params)
    # restore inveptionV3
    saver = tf.train.Saver(slim.get_model_variables())

    all_vars = [var for var in tf.global_variables()]
     
    all_vars = tf.global_variables()

    #model_vars = [k for k in all_vars if k.name.startswith('sc1')]

    # name of variable `my_var:0` corresponds `my_var` for loader
    #model_keys = [s.name.replace('sc1', 'InceptionV3')[:-2] for s in model_vars]

    #saver = tf.train.Saver(dict(zip(model_keys, model_vars)))


    print(len(all_vars))
    model_vars = [var for var in slim.get_model_variables()]
    print(len(model_vars))
    # if restore_vars == model_vars:
    # print("They are matched!!")

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver.restore(sess, FLAGS.checkpoint_path)

    for filenames, images in load_images(FLAGS.input_dir, batch_shape):
      print(filenames)
      target_class_for_batch = (
                       [all_images_taget_class[n] for n in filenames] + [0] * (FLAGS.batch_size - len(filenames)))
      adv_images = sess.run(x_adv, feed_dict={x_input: images, target_class_input: target_class_for_batch})
      save_images(adv_images, filenames, FLAGS.output_dir)
    
    sess.close()


if __name__ == '__main__':
  tf.app.run()
