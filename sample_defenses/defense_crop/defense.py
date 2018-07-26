"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from io import BytesIO
import random

import numpy as np
from scipy.misc import imread, imsave
import PIL
import PIL.Image

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
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'image_resize', 331, 'Resize of image size.')

FLAGS = tf.flags.FLAGS

DEBUG = False

def padding_layer_iyswim(inputs, shape, name=None):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    input_shape = tf.shape(inputs)
    input_short = tf.reduce_min(input_shape[1:3])
    input_long = tf.reduce_max(input_shape[1:3])
    output_long = tf.to_int32(tf.ceil(
        1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short)))
    output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_short
    output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_long
    return tf.pad(inputs, tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)


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
        with tf.gfile.Open(filepath, 'rb') as f:
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

def defend_crop(x, crop_size=90, ensemble_size=30):
    x_size = tf.to_float(x.shape[1])
    frac = crop_size/x_size
    start_fraction_max = (x_size - crop_size)/x_size
    def randomizing_crop(x):
        start_x = tf.random_uniform((), 0, start_fraction_max)
        start_y = tf.random_uniform((), 0, start_fraction_max)
        return tf.image.crop_and_resize(x, boxes=[[start_y, start_x, start_y+frac, start_x+frac]],
                                 box_ind=[0], crop_size=[crop_size, crop_size])

    return tf.concat([randomizing_crop(x) for _ in range(ensemble_size)], axis=0)

def main(_):
    batch_shape = [1, FLAGS.image_height, FLAGS.image_width, 3] # now let's work on one image per time
    num_classes = 1001
    itr = 30

    tf.logging.set_verbosity(tf.logging.INFO)
    print("input dir = {0}".format(FLAGS.input_dir))
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        #print('x_input.shape = {0}'.format(x_input.shape))
        cropped_xs = defend_crop(x_input)
        #print('after crop, x_input.shape = {0}'.format(cropped_xs.shape))
        #finished input transformation
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            cropped_logits, end_points_xs = inception_resnet_v2.inception_resnet_v2(
                cropped_xs, num_classes=num_classes, is_training=False, create_aux_logits=False)
            cropped_probs = tf.reduce_mean(tf.nn.softmax(cropped_logits), axis=0, keep_dims=True)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    #print("filename = {0}".format(filenames))
                    #final_preds = np.zeros([FLAGS.batch_size, num_classes, itr])
                    #print('min = {0}'.format(np.amin(images)))
                    #print('max = {0}'.format(np.amax(images)))
                    #print('after jpeg min = {0}'.format(np.amin(images)))
                    #print('after jpeg max = {0}'.format(np.amax(images)))
                    #print('images.shape after = {0}'.format(images.shape))
                    """
                    pred, aux_pred = sess.run([end_points_xs['Predictions'], end_points_xs['AuxPredictions']],
                                              feed_dict={x_input: images})
                    final_probs = pred + 0.4 * aux_pred
                    labels = np.argmax(final_probs, 1)
                    """
                    probs = sess.run(cropped_probs, feed_dict={x_input:images})

                    labels = np.argmax(probs, 1)

                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))

if __name__ == '__main__':
    tf.app.run()
