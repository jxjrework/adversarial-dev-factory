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

from tensorflow.contrib.slim.nets import inception

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

tf.flags.DEFINE_integer(
    'image_resize', 331, 'Resize of image size.')

FLAGS = tf.flags.FLAGS

DEBUG = False

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


def defend_reduce(arr, depth=3):
    """
    bit depth defense
    :param arr: numpy array
    :param depth:
    :return: denoised image
    """
    arr = (arr * 255.0).astype(np.uint8)
    shift = 8 - depth
    arr = (arr >> shift) << shift
    arr = arr.astype(np.float32)/255.0
    return arr


def bregman(image, mask, weight, eps=1e-3, max_iter=100):
    rows, cols, dims = image.shape

    rows2 = rows + 2
    cols2 = cols + 2
    total = rows * cols * dims
    shape_ext = (rows2, cols2, dims)

    u = np.zeros(shape_ext)
    dx = np.zeros(shape_ext)
    dy = np.zeros(shape_ext)
    bx = np.zeros(shape_ext)
    by = np.zeros(shape_ext)

    u[1:-1, 1:-1] = image
    # reflect image
    u[0, 1:-1] = image[1, :]
    u[1:-1, 0] = image[:, 1]
    u[-1, 1:-1] = image[-2, :]
    u[1:-1, -1] = image[:, -2]

    i = 0
    rmse = np.inf
    lam = 2 * weight
    norm = (weight + 4 * lam)

    while i < max_iter and rmse > eps:
        rmse = 0

        for k in range(dims):
            for r in range(1, rows+1):
                for c in range(1, cols+1):
                    uprev = u[r, c, k]

                    # forward derivatives
                    ux = u[r, c+1, k] - uprev
                    uy = u[r+1, c, k] - uprev

                    # Gauss-Seidel method
                    if mask[r-1, c-1]:
                        unew = (lam * (u[r+1, c, k] +
                                      u[r-1, c, k] +
                                      u[r, c+1, k] +
                                      u[r, c-1, k] +
                                      dx[r, c-1, k] -
                                      dx[r, c, k] +
                                      dy[r-1, c, k] -
                                      dy[r, c, k] -
                                      bx[r, c-1, k] +
                                      bx[r, c, k] -
                                      by[r-1, c, k] +
                                      by[r, c, k]
                                     ) + weight * image[r-1, c-1, k]
                               ) / norm
                    else:
                        # similar to the update step above, except we take
                        # lim_{weight->0} of the update step, effectively
                        # ignoring the l2 loss
                        unew = (u[r+1, c, k] +
                                  u[r-1, c, k] +
                                  u[r, c+1, k] +
                                  u[r, c-1, k] +
                                  dx[r, c-1, k] -
                                  dx[r, c, k] +
                                  dy[r-1, c, k] -
                                  dy[r, c, k] -
                                  bx[r, c-1, k] +
                                  bx[r, c, k] -
                                  by[r-1, c, k] +
                                  by[r, c, k]
                                 ) / 4.0
                    u[r, c, k] = unew

                    # update rms error
                    rmse += (unew - uprev)**2

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    # d_subproblem
                    s = ux + bxx
                    if s > 1/lam:
                        dxx = s - 1/lam
                    elif s < -1/lam:
                        dxx = s + 1/lam
                    else:
                        dxx = 0
                    s = uy + byy
                    if s > 1/lam:
                        dyy = s - 1/lam
                    elif s < -1/lam:
                        dyy = s + 1/lam
                    else:
                        dyy = 0

                    dx[r, c, k] = dxx
                    dy[r, c, k] = dyy

                    bx[r, c, k] += ux - dxx
                    by[r, c, k] += uy - dyy

        rmse = np.sqrt(rmse / total)
        i += 1

    return np.squeeze(np.asarray(u[1:-1, 1:-1]))


def defend_tv(input_array, keep_prob=0.5, lambda_tv=0.03):
    mask = np.random.uniform(size=input_array.shape[:2])
    mask = mask < keep_prob
    return bregman(input_array, mask, weight=2.0/lambda_tv)


def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    itr = 30
    tf.logging.set_verbosity(tf.logging.INFO)
    #print("input dir = {0}".format(FLAGS.input_dir))
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        #finished input
        with slim.arg_scope(inception.inception_v3_arg_scope()):
          _, end_points = inception.inception_v3(
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
                    for idxx in range(batch_shape[0]):
                        images[idxx, :, :, :] = defend_tv(images[idxx]) # tv defense
                    labels = sess.run(predicted_labels, feed_dict={x_input: images})
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
    tf.app.run()







# end of code
