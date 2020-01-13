# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
channelsNumDict = {
    0.5: [48, 1024], 
    1.0: [116, 1024], 
    1.5: [176, 1024], 
    2.0: [244, 2048]
    }
batch_norm_params = {
    'scale': True,
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
}
def channel_shuffle(inputs,groups=2,scope = 'channel_shuffle'):
    # the same operation with channel_permutation
    net = inputs
    N, H, W, C =net.get_shape().as_list()
    assert 0 == C % groups, "Input channels must be a multiple of groups"
    with tf.variable_scope(scope):
        net_reshaped = tf.reshape(net,[-1, H, W, groups, C // groups]) # 先合并重组
        net_transposed = tf.transpose(net_reshaped, [0, 1, 2, 4, 3])  # 转置
        net = tf.reshape(net_transposed, [-1, H, W, C])  # 摊平
        
        return net 
def shuffle_block_v1(inputs,channels,stride,groups,depth_multiplier=1,scope=None):
    
    with tf.variable_scope(scope): 
        prev_net = slim.separable_conv2d(inputs, None, [3, 3],
                                        depth_multiplier=depth_multiplier,
                                        stride=stride,
                                        activation_fn=None,
                                        scope='DWconv_skip')
        prev_net = slim.conv2d(prev_net,channels,1,1,scope = 'conv0/1x1')
        net = slim.conv2d(inputs,channels,1,1,scope = 'conv1/1x1')
        net = slim.separable_conv2d(net, None, [3, 3],
                                    depth_multiplier=depth_multiplier,
                                    stride=stride,
                                    activation_fn=None,
                                    scope='DWconv')
        net = slim.conv2d(net,channels,1,1,scope = 'conv2/1x1')
        net = tf.concat([prev_net,net], axis=3,name='concat')
        net = channel_shuffle(net,groups, scope="channelchuffle")
            
        return net        
    
def shuffle_block_v2(inputs,channels,stride,groups,depth_multiplier=1,scope=None):
    
    net = inputs
    assert channels % 2 == 0
    channels = channels//2
    with tf.variable_scope(scope): 
        net_split = tf.split(net,2,axis=3)
        net = slim.conv2d(net_split[1],channels,1,1,scope = 'conv1/1x1')
        net = slim.separable_conv2d(net, None, [3, 3],
                                    depth_multiplier=depth_multiplier,
                                    stride=stride,
                                    activation_fn=None,
                                    scope='DWconv')
        net = slim.conv2d(net,channels,1,1,scope = 'conv2/1x1')
        net = tf.concat([net_split[0],net], axis=3,name='concat')
        net = channel_shuffle(net,groups, scope="channelchuffle")
        
        return net
        
            
def shuffle_bottleneck(inputs,channels,repeat,stride,groups,depth_multiplier=1,scope=None):
    with tf.variable_scope(scope):
        net=inputs
        if stride == 2:
            net=shuffle_block_v1(net,channels//2,stride,groups,depth_multiplier,scope='shufflebk_0')
            stride = 1
            print('shufflebk_0:',net)
        for i in range(repeat-1):
            net=shuffle_block_v2(net,channels,stride,groups,depth_multiplier,scope='shufflebk_'+str(i+1))
            print('shufflebk_%d:'%(i+1),net)
        return net
        
        
def shuffle_net(inputs,channelsNum,groups,depth_multiplier,scope=None):
    net = inputs
    with tf.variable_scope(scope): 
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d,slim.max_pool2d], padding='SAME'):
            net = slim.conv2d(net,24,3,2,scope = 'conv0')       #input:224x224
            print('conv0: ',net)
            net = slim.max_pool2d(net, 3, 2,scope='maxpool1')   #input:112x112
            print('maxpool1:',net)
            net = shuffle_bottleneck(net,channelsNum[0],4,2,groups,depth_multiplier,scope='stage2')
            net = shuffle_bottleneck(net,channelsNum[0]*2,8,2,groups,depth_multiplier,scope='stage3')
            net = shuffle_bottleneck(net,channelsNum[0]*4,4,2,groups,depth_multiplier,scope='stage4')
            net = slim.conv2d(net,channelsNum[1],1,1,scope = 'conv5')
            net=slim.avg_pool2d(net,net.get_shape()[1:3],padding='VALID',scope='AvgPool')
            net=tf.squeeze(net,[1,2])
        return net 
def shufflenet_cls(inputs,
                channelsNum,
                is_training=True,
                embedding_size=128,
                groups=2,
                depth_multiplier=1,
                reuse=None,
                scope=None):
        input_shape = inputs.get_shape().as_list()
        net =inputs
        if len(input_shape) != 4:
            raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))
        with tf.variable_scope(scope, 'ShuffleNet_v2', [inputs], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm], is_training=is_training): 
                net = shuffle_net(net,channelsNum,groups,depth_multiplier,scope='shufflenet')
                if embedding_size:
                    net = slim.fully_connected(net, embedding_size, activation_fn=None,scope='Logits', reuse=False)  
            return net 
def shufflenet_arg_scope(weight_decay=0.0005,use_batch_norm=True):
    """Defines the default MobilenetV2 arg scope.
    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
    Returns:
      An `arg_scope` to use for the shufflenet v2 model.
    """
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
    #weights_initializer = slim.xavier_initializer_conv2d()
    with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.separable_conv2d],
                            weights_initializer=weights_initializer,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_regularizer=slim.l2_regularizer(weight_decay)) :
        with slim.arg_scope([slim.conv2d,slim.separable_conv2d], 
                        activation_fn=tf.nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        biases_initializer=None) as scope:
            return scope
def inference(inputs,
              embedding_size=128,
              keep_probability=0.999,
              weight_decay=0.0005,
              use_batch_norm=True,
              phase_train=True,
              reuse=None,
              scope='ShuffleNet_v2'):
    arg_scope = shufflenet_arg_scope(weight_decay,use_batch_norm)
    channelsNum = channelsNumDict[1.5]
    with slim.arg_scope(arg_scope):
        net = shufflenet_cls(inputs,
                            channelsNum,
                            is_training=phase_train,
                            embedding_size=embedding_size,
                            groups=2,
                            depth_multiplier=1,
                            reuse=reuse,
                            scope=scope)
    return net, None                