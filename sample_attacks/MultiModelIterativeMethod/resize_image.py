import pandas as pd
import tensorflow as tf

print('hello world')

tf.flags.DEFINE_string(
    'path', '', 'The address of the TensorFlow master to use.')

FLAGS = tf.flags.FLAGS

#df = pd.read_csv(FLAGS.path)
#print(df.head())

from PIL import Image
# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder
imageFile = FLAGS.path
im1 = Image.open(imageFile)
# adjust width and height to your needs
width = 299
height = 299
# use one of these filter options to resize the image
im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
ext = ".png"
im2.save(imageFile[:-4] + '_resized_' + ext)