#coding=utf-8
"""
MobileNet v2.

As described in https://arxiv.org/abs/1801.04381

  Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict

import tensorflow as tf
import tensorflow.contrib.slim as slim

batch_norm_params = {
    'scale': True,
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.00001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
}

Conv = namedtuple('Conv', ['kernel', 'stride', 'channel'])
InvertedBottleneck = namedtuple('InvertedBottleneck', ['up_sample', 'channel', 'stride', 'repeat'])
AvgPool = namedtuple('AvgPool', ['kernel'])

# Sequence of layers, described in Table 2
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, channel=32),              # first block, input 224x224x3
    InvertedBottleneck(up_sample=1, channel=16, stride=1, repeat=1),  # second block, input : 112x112x32
    InvertedBottleneck(up_sample=6, channel=24, stride=2, repeat=2),  # third block, input: 112x112x16
    InvertedBottleneck(up_sample=6, channel=32, stride=2, repeat=3),  # fourth block, input: 56x56x24
    InvertedBottleneck(up_sample=6, channel=64, stride=2, repeat=4),  # fifth block, input: 28x28x32
    InvertedBottleneck(up_sample=6, channel=96, stride=1, repeat=3),  # sixth block, input: 28x28x64
    InvertedBottleneck(up_sample=6, channel=160, stride=2, repeat=3),  # seventh block, input: 14x14x96
    InvertedBottleneck(up_sample=6, channel=320, stride=1, repeat=1),  # eighth block, input: 7x7x160
    Conv(kernel=[1, 1], stride=1, channel=1280),
    AvgPool(kernel=[7, 7])
]

def mobilenet_v2_base(inputs,
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      scope=None):
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    normalize_channels = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = OrderedDict()

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    net = inputs
    with tf.variable_scope(scope, 'MobilenetV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            for i, conv_def in enumerate(conv_defs):
                end_point = ''
                if isinstance(conv_def, Conv):
                    end_point = 'Conv2d_%d' % i
                    num_channel = normalize_channels(conv_def.channel)
                    net = slim.conv2d(net, num_channel, conv_def.kernel,
                                      activation_fn=tf.nn.relu,
                                      stride=conv_def.stride,
                                      scope=end_point)
                    end_points[end_point] = net
                elif isinstance(conv_def, InvertedBottleneck):
                    stride = conv_def.stride

                    if conv_def.repeat <= 0:
                        raise ValueError('repeat value of inverted bottleneck should be greater than zero.')

                    for j in range(conv_def.repeat):
                        end_point = 'InvertedBottleneck_%d_%d' % (i, j)
                        prev_output = net
                        net = slim.conv2d(net, conv_def.up_sample * net.get_shape().as_list()[-1], [1, 1],
                                          activation_fn=tf.nn.relu,
                                          scope=end_point + '_inverted_bottleneck')
                        end_points[end_point + '_inverted_bottleneck'] = net
                        net = slim.separable_conv2d(net, None, [3, 3],
                                                    depth_multiplier=1,
                                                    stride=stride,
                                                    activation_fn=tf.nn.relu,
                                                    scope=end_point + '_dwise')
                        end_points[end_point + '_dwise'] = net

                        num_channel = normalize_channels(conv_def.channel)
                        net = slim.conv2d(net, num_channel, [1, 1],
                                          activation_fn=None,
                                          scope=end_point + '_linear')
                        end_points[end_point + '_linear'] = net

                        if stride == 1:
                            if prev_output.get_shape().as_list()[-1] != net.get_shape().as_list()[-1]:
                                # Assumption based on previous ResNet papers: If the number of filters doesn't match,
                                # there should be a conv 1x1 operation.
                                # reference(pytorch) : https://github.com/MG2033/MobileNet-V2/blob/master/layers.py#L29
                                prev_output = slim.conv2d(prev_output, num_channel, [1, 1],
                                                          activation_fn=None,
                                                          biases_initializer=None,
                                                          scope=end_point + '_residual_match')

                            # as described in Figure 4.
                            net = tf.add(prev_output, net, name=end_point + '_residual_add')
                            end_points[end_point + '_residual_add'] = net
                        stride = 1
                elif isinstance(conv_def, AvgPool):
                    end_point = 'AvgPool'
                    net = slim.avg_pool2d(net, conv_def.kernel, scope=end_point)
                    net = slim.flatten(net, scope='Flatten')
                    end_points[end_point] = net
                else:
                    raise ValueError('CONV_DEF is not valid.')

    return net, end_points


def mobilenet_v2_cls(inputs,
                     is_training=True,
                     min_depth=8,
                     depth_multiplier=1.0,
                     conv_defs=None,
                     embedding_size=128,
                     reuse=None,
                     scope='MobilenetV2'):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v2_base(inputs, scope=scope,
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier,
                                                conv_defs=conv_defs)
            with tf.variable_scope('preLogits'):
                if embedding_size:
                    net = slim.fully_connected(net,embedding_size, activation_fn=None, scope='FC', reuse=False)
                    end_points['FC'] = net

        return net, end_points


def mobilenet_v2_arg_scope(weight_decay=0.00004, use_batch_norm=True):
    """Defines the default MobilenetV2 arg scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
    Returns:
      An `arg_scope` to use for the mobilenet v2 model.
    """
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}

    #weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
    weights_initializer = slim.xavier_initializer_conv2d()
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected], weights_initializer=weights_initializer):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        biases_initializer=None):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer)as sc:
                    return sc
