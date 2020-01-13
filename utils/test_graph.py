#matplotlib inline

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE

from flip_gradient import flip_gradient
from utils import *
import os
import datasets
import importlib
import logging
from datetime import datetime
from tensorflow.python import pywrap_tensorflow
tf.logging.set_verbosity(tf.logging.INFO)
import utils_tf
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

flags = tf.app.flags
flags.DEFINE_string('checkpoint_path', './pretrained_models/20190416_021402_l2_softmax.ckpt-43000',
                    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_boolean('restore_from_base_network', False,
                     'Restore model only for base network.')
flags.DEFINE_string('save_dir', './trained_models/',
                    'The path to save a checkpoint and model file.')
                    
FLAGS = flags.FLAGS

def _model_restore_fn(ckpt_path, sess, default_to_restore, restore_from_base_network=False):
    if restore_from_base_network:
        tf.logging.info('Load pretrained model in {}'.format(ckpt_path))
        model_variables = slim.get_model_variables()
        def get_variable_name(var):
            return var.op.name
        restore_variables_dict = {}
        for var in model_variables:
            var_name = get_variable_name(var)
            #var_name = var_name.replace('CNNBase', 'WQ_V3', 1)
            restore_variables_dict[var_name] = var
        print ('-----------------------------------')
        print (len(restore_variables_dict))
        for key, item in restore_variables_dict.items():
            tf.logging.info('Restore variable: {}'.format(item.name))
            
        restorer = tf.train.Saver(restore_variables_dict)
        restorer.restore(sess, ckpt_path)
        print ('restore done')
    else:
        tf.logging.info('Load pretrained model in {}'.format(ckpt_path))
        for label in default_to_restore:
            print (len(default_to_restore), label)
            #for name in default_to_restore[label]:
            #    records.append('{},{}'.format(name, label))
        restorer = tf.train.Saver(var_list=default_to_restore)
        restorer.restore(sess, ckpt_path)

def normalized_linear_layer(inputs, 
                            num_outputs,
                            weights_initializer, 
                            weights_regularizer,
                            scale_inputs=None,
                            normalize_weights=True,
                            reuse=None,
                            trainable=True, 
                            name=None):
    """Linear transform layer with inputs or weights normalization.

    No bias term and no activation function.

    Args:
        inputs: Must be 2D tensor.
        num_outputs: Integer, the number of output units in the layer.
        weights_initializer: An initializer for the weights.
        weights_regularizer: An regularizer for the weights.
        scale_inputs: If scale_inputs is None, no normalization on inputs.
            If scale_inputs is not None, normalization on inputs and scale it.
        normalize_weights: Whether or not normalize weights.
        reuse: Whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        name: Optional scope for variable_scope.
    """
    with tf.variable_scope(name, 'normalized_linear_layer', [inputs], reuse=reuse):
        assert inputs.get_shape().ndims == 2, "Inputs Tensor shape must be 2-D"
        num_inputs = inputs.get_shape().as_list()[-1]
        weights = tf.get_variable('weights',
                                  shape=[num_inputs, num_outputs],
                                  initializer=weights_initializer,
                                  regularizer=weights_regularizer,
                                  trainable=trainable,
                                  dtype=inputs.dtype)
        if normalize_weights:
            weights = tf.nn.l2_normalize(weights, 0, 1e-12)

        if scale_inputs is None:
            outputs = tf.matmul(inputs, weights)
        else:
            inputs_normalized = tf.nn.l2_normalize(inputs, 1, 1e-12)
            cos_theta = tf.matmul(inputs_normalized, weights)
            outputs = scale_inputs * cos_theta
        return outputs
        
class shufflenet_v2(object):
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.int32, [None])
        self.label_y = tf.one_hot(self.y, 20000)
        self.classify_labels = self.label_y
        self.train = tf.placeholder(tf.bool, [])
        self.l = tf.placeholder(tf.float32, [])
        self.scale = 4.5
        self.weight_decay = 0.0005
        self.num_classes = 20000
        
        model_def = 'models.ShuffleNet_v2'
        network = importlib.import_module(model_def)
        self.prelogits, endpoints = network.inference(self.X,
                                    embedding_size = 128,
                                    keep_probability = 0.6,
                                    weight_decay = self.weight_decay,
                                    use_batch_norm = True,
                                    phase_train = True,
                                    scope = None)
                                    
        self.scale_inner = tf.get_variable('Logits/scale', (), dtype=self.prelogits.dtype,
            initializer=tf.constant_initializer(self.scale), 
            regularizer=slim.l2_regularizer(self.weight_decay))
                                    
        logits = normalized_linear_layer(self.prelogits, self.num_classes, 
                        weights_initializer=tf.truncated_normal_initializer(stddev=1e-1),
                        weights_regularizer=slim.l2_regularizer(self.weight_decay),
                        scale_inputs=self.scale_inner,
                        normalize_weights=False,
                        name='Logits')
                                    
        self.pred = tf.nn.softmax(logits)
        self.pred_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y, logits=logits), name='l2_softmax_loss')
        
#model_s = shufflenet_v2()
#logits = model_s.prelogits

default_to_restore = tf.trainable_variables()
#default_to_restore += tf.get_collection(tf.GraphKeys.RESTORE_VARIABLES)
sess = tf.Session(config=utils_tf.get_default_sess_config())
if FLAGS.checkpoint_path:
    _model_restore_fn(
        FLAGS.checkpoint_path, sess, default_to_restore, FLAGS.restore_from_base_network)
                
# print ('load model----------------------------')
# pretrained_model = utils_tf.normalize_pretrained_model(FLAGS.checkpoint_path)
# model_restore_func = None

# default_to_restore = tf.trainable_variables()
# default_to_restore += tf.get_collection(tf.GraphKeys.RESTORE_VARIABLES)
# saver = tf.train.Saver(default_to_restore, max_to_keep=3)
# sess = tf.Session(config=utils_tf.get_default_sess_config())
# loader = utils_tf.restore(sess, pretrained_model, default_to_restore, model_restore_func)

# sess.graph.finalize()
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord, sess=sess)