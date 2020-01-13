#coding=utf-8
# Copyright (c) 2017-2018 Dilusense Inc. All Rights Reserved.

import tensorflow as tf

def dap_func_wrapper(image_height, image_width):
    """ DAP, decode, augment and preprocess
        先进行数据增强，再进行预处理
    """
    def dap_func(content, augment=False):
        # decode
        image = tf.image.decode_image(content, channels=3)
        image = tf.cast(image, tf.float32)
        
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
        image -= 127.5
        image /= 128.0
        
        # set shape, required
        image.set_shape([image_height, image_width, 3])
        
        return image
        
    return dap_func
    