# coding=utf-8
"""
Copyright (c) 2017-2018 Dilusense Inc. All Rights Reserved.

@author: hucs and kangkai
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from .mobilenet_v2_relu_bases import mobilenet_v2_arg_scope
from .mobilenet_v2_relu_bases import mobilenet_v2_cls

def inference(inputs,
              embedding_size=128,
              keep_probability=0.999,
              weight_decay=0.00004,
              use_batch_norm=True,
              phase_train=True,
              reuse=None,
              scope='MobilenetV2'):
    arg_scope = mobilenet_v2_arg_scope(weight_decay, use_batch_norm)
    with slim.arg_scope(arg_scope):
        net, end_points = mobilenet_v2_cls(inputs,
                                           is_training=phase_train,
                                           min_depth=8,
                                           depth_multiplier=0.75,
                                           conv_defs=None,
                                           embedding_size=embedding_size,
                                           reuse=reuse,
                                           scope=scope)
    """
    names = []
    for var in tf.trainable_variables():
        names.append(var.op.name)
    names = sorted(names)
    for k, name in enumerate(names):
        print('[{}/{}] {}'.format(k+1, len(names), name))
    """
    return net, end_points
