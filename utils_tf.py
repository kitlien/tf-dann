#coding=utf-8
"""
Copyright (c) 2017-2018 Dilusense Inc. All Rights Reserved.

@author: kangkai and chenzc
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import importlib
from collections import OrderedDict

import numpy as np
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph

def get_model_params(model_weight, var_names=None, only_shape=False, sort=False):
    """Returns an OrderedDict of params indexed by name from the checkpoint.

    References:
        `tf.train.list_variables` or `tf.contrib.framework.list_variables`.
        caffe.Net.params in PyCaffe
    """
    if tf.gfile.IsDirectory(model_weight):
        model_weight = tf.train.latest_checkpoint(model_weight)
    if model_weight is None:
        raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                     "given directory %s" % model_weight)
    reader = tf.train.NewCheckpointReader(model_weight)
    
    variable_map = reader.get_variable_to_shape_map()
    if var_names is None:
        var_names = variable_map.keys()
    if sort is True:
        var_names = sorted(var_names)
        
    params_dict = OrderedDict()
    if only_shape is True:
        for var_name in var_names:
            params_dict[var_name] = variable_map[var_name]
    else:
        for var_name in var_names:
            params_dict[var_name] = reader.get_tensor(var_name)
    return params_dict

def build_inference(model_def, image_batch, embedding_size, keep_probability, 
                weight_decay, use_batch_norm, phase_train, scope=None):
    """Build inference graph.
    """
    if isinstance(model_def, str):
        network = importlib.import_module(model_def)
        outputs, endpoints = network.inference(image_batch,
                                    embedding_size=embedding_size,
                                    keep_probability=keep_probability,
                                    weight_decay=weight_decay,
                                    use_batch_norm=use_batch_norm,
                                    phase_train=phase_train,
                                    scope=scope)
    elif hasattr(model_def, '__call__'):
        outputs, endpoints = model_def(image_batch,
                                    embedding_size=embedding_size,
                                    keep_probability=keep_probability,
                                    weight_decay=weight_decay,
                                    use_batch_norm=use_batch_norm,
                                    phase_train=phase_train,
                                    scope=scope)
    else:
        raise ValueError("Unrecognized model_def: should be string or function. "
                "Got {}.".format(str(model_def)))
    return outputs, endpoints
    
def build_inference_for_dream(model_def, image_batch, y_batch, embedding_size, keep_probability, 
                weight_decay, use_batch_norm, phase_train):
    """Build inference graph for Dream Block(Multi inputs).
    """
    if isinstance(model_def, str):
        network = importlib.import_module(model_def)
        outputs, endpoints = network.inference(image_batch, y_batch, 
                                    embedding_size=embedding_size,
                                    keep_probability=keep_probability,
                                    weight_decay=weight_decay,
                                    use_batch_norm=use_batch_norm,
                                    phase_train=phase_train)
    elif hasattr(model_def, '__call__'):
        outputs, endpoints = model_def(image_batch, y_batch, 
                                    embedding_size=embedding_size,
                                    keep_probability=keep_probability,
                                    weight_decay=weight_decay,
                                    use_batch_norm=use_batch_norm,
                                    phase_train=phase_train)
    else:
        raise ValueError("Unrecognized model_def: should be string or function. "
                "Got {}.".format(str(model_def)))
    return outputs, endpoints
    
def get_trainable_var_names(model_def, image_shape, embedding_size=None, 
                            use_batch_norm=False, dtype=tf.float32):
    tf.reset_default_graph()
    image_shape = list(image_shape)
    assert len(image_shape) in (2, 3)
    if len(image_shape) == 2:
        image_shape += [1]
    h, w, c = image_shape
    images_placeholder = tf.placeholder(dtype, shape=(None, h, w, c))
    build_inference(model_def, images_placeholder, 
                    embedding_size=embedding_size, keep_probability=1.0, 
                    weight_decay=0.0, use_batch_norm=use_batch_norm, 
                    phase_train=False)
    var_names = []
    for var in tf.trainable_variables():
        var_names.append(var.op.name)
    tf.reset_default_graph()
    return var_names
    
def normalize_pretrained_model(pretrained_model):
    """
    References:
        `_get_checkpoint_filename` in python/training/checkpoint_utils.py
    """
    # pretrained_model maybe None
    if pretrained_model:
        pretrained_model = os.path.expanduser(pretrained_model)
        if tf.gfile.IsDirectory(pretrained_model):
            pretrained_model = tf.train.latest_checkpoint(pretrained_model)
    return pretrained_model
    
def get_default_sess_config():
    """
    References:
        tensorflow-master/tensorflow/core/protobuf/config.proto
    """
    config = tf.ConfigProto()
    # If true, the allocator does not pre-allocate the entire specified
    # GPU memory region, instead starting small and growing as needed.
    config.gpu_options.allow_growth = True
    # Start running operations on the Graph. allow_soft_placement must be set to True
    # to build towers on GPU, as some of the ops do not have GPU implementations.
    # Whether soft placement is allowed. If allow_soft_placement is true,
    # an op will be placed on CPU if one of the following is true:
    #   1. there's no GPU implementation for the OP
    #   2. no GPU devices are known or registered
    #   3. need to co-locate with reftype input(s) which are from CPU.
    config.allow_soft_placement = True    
    return config
    
def guarantee_initialized_variables(sess, var_list=None):
    """
    References:
        https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
        https://stackoverflow.com/questions/44268206/how-to-get-the-list-of-uninitialized-variables-from-tf-report-uninitialized-vari/44276281
    """
    if var_list is None:
        var_list = tf.global_variables() + tf.local_variables()
    varnames = sess.run(tf.report_uninitialized_variables(var_list))
    uninitialized_vars = [v for v in var_list if v.name.split(':')[0] in set(varnames)]
    sess.run(tf.variables_initializer(uninitialized_vars))
    return uninitialized_vars
    
def restore(sess, pretrained_model, vars_to_restore, model_restore_func=None):
    """Restore network parameter from pretrained model.
    
    The model_restore_func with following signature: 
        `def model_restore_func(sess, saver, pretrained_model)` with no return value or
        `def model_restore_func(pretrained_model)` which return a list of `Variable`/`SaveableObject`, 
            or a dictionary mapping names to `SaveableObject`s.
    the second signature is preferred.
        
    For example:
    ```
    def model_restore_func(sess, saver, pretrained_model):
        reader = tf.train.NewCheckpointReader(pretrained_model)
        var_names = reader.get_variable_to_shape_map().keys()
        for k, var in enumerate(tf.trainable_variables()):
            print('[{}/{}] {}'.format(k+1, len(tf.trainable_variables()), var.op.name))
            if var.op.name.startswith('DenseNet/Bottleneck') or var.op.name.startswith('Logits'):
                continue
            # you can change dict's key (var.op.name) according to pretrained_model
            if var.op.name in var_names:
                var.load(reader.get_tensor(map_func(var.op.name)), sess)
    ```
    and
    ```
    def model_restore_func(pretrained_model):
        reader = tf.train.NewCheckpointReader(pretrained_model)
        var_names = reader.get_variable_to_shape_map().keys()
        var_dict = {}
        for k, var in enumerate(tf.trainable_variables()):
            #print('[{}/{}] {}'.format(k+1, len(tf.trainable_variables()), var.op.name))
            if var.op.name.startswith('DenseNet/Bottleneck') or var.op.name.startswith('Logits'):
                continue
            # you can change dict's key (var.op.name) according to pretrained_model
            if var.op.name in var_names:
                var_dict.update({map_func(var.op.name): var})
        return var_dict
    ```
    
    References:
        caffe.Net.copy_from in PyCaffe
        Trainer.restore in tflearn/helpers/trainer.py
    """
    pretrained_model = normalize_pretrained_model(pretrained_model)
    
    try:
        if (pretrained_model is not None) and (model_restore_func is not None):
            vars_to_restore = model_restore_func(pretrained_model)
            
        loader = tf.train.Saver(var_list=vars_to_restore)
        with sess.as_default():
            start_time = time.time()
            # Some variables maybe not restored, so initialize all uninitialized variables before restoration.
            guarantee_initialized_variables(sess, None)
            if pretrained_model:
                print('Restoring pretrained model: {}'.format(pretrained_model))
                loader.restore(sess, pretrained_model)
                print('Model restored in {:.2f} seconds'.format(time.time() - start_time))
            else:
                print('Model random initialization in {:.2f} seconds'.format(time.time() - start_time))
    except:
        loader = tf.train.Saver(var_list=vars_to_restore)
        with sess.as_default():
            start_time = time.time()
            # Some variables maybe not restored, so initialize all uninitialized variables before restoration.
            guarantee_initialized_variables(sess, None)
            if pretrained_model is not None:
                print('Restoring pretrained model: {}'.format(pretrained_model))
                if model_restore_func is not None:
                    model_restore_func(sess, loader, pretrained_model)
                else:
                    loader.restore(sess, pretrained_model)
                print('Model restored in {:.2f} seconds'.format(time.time() - start_time))
            else:
                print('Model random initialization in {:.2f} seconds'.format(time.time() - start_time))
        
    return loader

def write_frozen_graph(filename, sess, output_node_names):
    output_graph_def = graph_util.convert_variables_to_constants(sess, 
        sess.graph_def, output_node_names)
    with tf.gfile.FastGFile(filename, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

def write_tflite(filename, inputs, outputs, sess, output_node_names):
    output_graph_def = graph_util.convert_variables_to_constants(sess, 
        sess.graph_def, output_node_names)
    tflite_model = tf.contrib.lite.toco_convert(output_graph_def, inputs, outputs)
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    
def read_graph(filename, binary=True, clear_devices=True):
    graph_def = tf.GraphDef()
    if binary:
        with tf.gfile.FastGFile(filename, 'rb') as f:
            graph_def.ParseFromString(f.read())
    else:
        with tf.gfile.FastGFile(filename, 'r') as f:
            text_format.Merge(f.read(), graph_def)
    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        for node in graph_def.node:
            node.device = ""
    return graph_def

def get_graph_op_types(graph, sort=True):
    op_list = []
    for op in graph.get_operations():
        op_list.append(op.type)
    op_list = list(set(op_list))
    if sort:
        op_list.sort() 
    return op_list
    
    