import os
import datasets
import importlib
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
#import utils_tf
import logging
import numpy as np
import cv2

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


flags = tf.app.flags
flags.DEFINE_string('checkpoint_path', './pretrained_models/20190416_021402_l2_softmax.ckpt-80000',
                    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_integer('batch_size', 96, 'Batch size.')
flags.DEFINE_boolean('restore_from_base_network', False,
                     'Restore model only for base network.')
flags.DEFINE_string('save_dir', './trained_models/',
                    'The path to save a checkpoint and model file.')
                    
FLAGS = flags.FLAGS

def dap_func(content, augment=False):
    # decode
    image = tf.image.decode_image(content, channels=3)
    image = tf.cast(image, tf.float32)
    image_height = 224
    image_width = 224
    # augment
    if augment:
        image = tf.random_crop(image, [image_height, image_width, 3])
        image = tf.image.random_flip_left_right(image)
        """
        noise = tf.random_normal([image_height, image_width, 1], mean=0.0, stddev=2, dtype=tf.float32)
        image = tf.minimum(tf.maximum(image + noise, 0), 255)
        """
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)

    # preprocess
    #image -= 127.5
    #image /= 128.0
    
    # set shape, required
    image.set_shape([image_height, image_width, 3])
    
    return image
dataset = datasets.datagate_dataset(is_training = False, 
                               batch_size = FLAGS.batch_size, 
                               input_height= 224, 
                               input_width= 224, 
                               input_channels= 3)
                               
for i in range(20000):
    image_name, X, y,iter = dataset.next_batch_source()
    if i%1000 == 0:
        print (i, image_name[0])
#name = tf.convert_to_tensor(np.array(image_name[0]))
# print (image_name[0])
# img = cv2.imread(image_name[0])
# cv2.imwrite("test.jpg",img)
# for i in range(10):
    # augment = np.random.randint(low = 0, high = 2)
    # print (augment)
    # if augment == 1:
        # print ("True")
        # img = cv2.flip(img,1)
        # cv2.imwrite("flip.jpg",img)

# content = tf.read_file(name)
# image = dap_func(content, augment = True)
# with tf.Session() as sess:
    # tf.global_variables_initializer().run()
    # print (sess.run(tf.shape(image)))
    #print (sess.run(image[0]))
    
# print (X[0][0])
# print (image_name)
