#coding=utf-8
"""
Copyright (c) 2017-2018 Dilusense Inc. All Rights Reserved.

@author: kangkai
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import importlib
from collections import OrderedDict

import numpy as np
import tensorflow as tf

def reduce_tensor_list(tensor_list, reduce_type='mean', name=None):
    assert isinstance(tensor_list, list)
    with tf.name_scope(name, "reduce_tensor_list", [tensor_list]) as name:
        if reduce_type == 'mean':
            return tf.reduce_mean(tensor_list, axis=0)
        elif reduce_type == 'sum':
            return tf.reduce_sum(tensor_list, axis=0)
        elif reduce_type == 'concat':
            return tf.concat(tensor_list, axis=0)
        else:
            raise ValueError('Invalid reduce_type')
        
def _get_tensor_shape(x, rank=None):
    """Returns the dimensions of a Tensor as list of integers or integer scalar tensors.
    Args:
        x: N-d Tensor;
        rank: Rank of the Tensor. If None, will try to guess it.
      
    Returns:
        A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
          input tensor.  Dimensions that are statically known are python integers,
          otherwise they are integer scalar tensors.
        
    References:
        https://github.com/balancap/SSD-Tensorflow/blob/master/tf_extended/tensors.py
        https://github.com/tensorflow/models/blob/master/research/object_detection/utils/shape_utils.py
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]
                
def get_tensor_shape(x, rank=None):
    """Returns the incoming data shape.
    """
    if isinstance(x, tf.Tensor):
        return _get_tensor_shape(x, rank)
    elif isinstance(x, (np.array, np.ndarray, list, tuple)):
        return np.shape(x)
    else:
        raise Exception("Invalid data type.")
        
def align_up_tensor(inputs, align_val, axis=0, name=None):
    """
    References:
        https://github.com/balancap/SSD-Tensorflow/blob/master/tf_extended/tensors.py
    """
    assert isinstance(align_val, int) and (align_val > 0)
    assert isinstance(inputs, tf.Tensor)
    
    with tf.name_scope(name, "align_up_tensor", [inputs]) as name:
        shape = get_tensor_shape(inputs)
        size = shape[axis]
        align_size = (size + align_val - 1) // align_val * align_val
        zeros = tf.zeros((tf.rank(inputs)-1, 2), dtype=tf.int32)
        padding = [[0, align_size - size]]
        paddings = tf.concat([zeros[:axis], padding, zeros[axis:]], axis=0)
        # When `mode='REFLECT'`, and `size=1`, it crashed with the following error:
        # InvalidArgumentError (see above for traceback): paddings must be 
        # less than the dimension size: 0, 1 not less than 1
        aligned_tensor = tf.pad(inputs, paddings, mode='SYMMETRIC')
        shape[axis] = align_size
        aligned_tensor = tf.reshape(aligned_tensor, tf.stack(shape))
        return aligned_tensor
    
def build_optimizer(optimizer, learning_rate):
    """
    Args:
        optimizer: Optimizer to use. Various ways of passing optimizers, include:
            - str, name of the optimizer like 'ADAGRAD', 'ADAM' and so on.
                E.g. `build_optimizer(optimizer='ADAM')`.
            - function, takes learning rate `Tensor` as argument and must return
                `Optimizer` instance. E.g. `build_optimizer(
                optimizer=lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.5))`.
                Alternatively, if `learning_rate` is `None`, the function takes no
                arguments. E.g. `build_optimizer(learning_rate=None,
                optimizer=lambda: tf.train.MomentumOptimizer(0.5, momentum=0.5))`.
            - class, subclass of `Optimizer` that takes only one required argument -
                learning rate, such as AdamOptimizer, AdagradOptimizer.
                E.g. `build_optimizer(optimizer=tf.train.AdagradOptimizer)`.
            - object, instance of subclass of `Optimizer`.
                E.g., `build_optimizer(optimizer=tf.train.AdagradOptimizer(0.5))`.
        learning_rate: A `Tensor` or a floating point value. The learning rate.
        
    References:
        tensorflow/contrib/layers/python/layers/optimizers.py
        tflearn/layers/estimator.py
    """
    if isinstance(optimizer, str):
        if learning_rate is None:
            raise ValueError("Learning rate is None, but should be specified if "
                            "optimizer is string ({}).".format(optimizer))
        if optimizer.upper() == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer.upper() == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer.upper() == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-8)
        elif optimizer.upper() == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
        elif optimizer.upper() == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer.upper() in ['MOM', 'MOMENT', 'MOMENTUM']:
            opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid Optimizer str: got {}'.format(optimizer))
    elif isinstance(optimizer, type) and issubclass(optimizer, tf.train.Optimizer):
        if learning_rate is None:
            raise ValueError("Learning rate is None, but should be specified if "
                            "optimizer is class ({}).".format(optimizer))
        opt = optimizer(learning_rate)
    elif hasattr(optimizer, '__call__'):
        if learning_rate is None:
            opt = optimizer()
        else:
            opt = optimizer(learning_rate)
        if not isinstance(opt, tf.train.Optimizer):
            raise ValueError("Unrecognized optimizer: function should return "
                            "subclass of Optimizer. Got {}.".format(str(opt)))
    elif isinstance(optimizer, tf.train.Optimizer):
        # NB: learning_rate is not used
        opt = optimizer
    else:
        raise ValueError("Unrecognized optimizer: should be string, "
                        "subclass of Optimizer, instance of "
                        "subclass of Optimizer or function with one argument. "
                        "Got {}.".format(str(optimizer)))
    return opt
    
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
    
def apply_loss_average(total_loss, decay, scope, log_scalars=True):
    # Compute the moving average of all individual losses and the total loss. 
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    loss_averages = tf.train.ExponentialMovingAverage(decay, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    if log_scalars:
        # Attach a scalar summmary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name +'_raw', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
        
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
        
    return total_loss
    
def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, '%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    
    metagraph_filename = os.path.join(model_dir, '%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.isfile(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    
def average_gradients(tower_grads_and_vars, name=None):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads_and_vars: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
          
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
        
    References:
        Adapted from `average_gradients` in `tensorflow-models-master/tutorials/image/cifar10`
    """
    with tf.name_scope(name, "average_gradients", tower_grads_and_vars) as name:
        average_grads = []
        for grads_and_vars in zip(*tower_grads_and_vars):
            # Note that each grads_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grads_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                # Append on a 'tower' dimension which we will average over below.
                if g is not None:
                    grads.append(tf.expand_dims(g, 0))
                    
            # In TF 1.2.0, if missing `if grads:`, it crashed with the following error:
            # ValueError: List argument 'values' to 'ConcatV2' Op with length 0 shorter than minimum length 2.
            if grads:
                # Average over the 'tower' dimension.
                grad = tf.concat(grads, axis=0)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So we will just return the first tower's pointer to the Variable.
                grad_and_var = (grad, grads_and_vars[0][1])
                average_grads.append(grad_and_var)
        
    return average_grads
    
def normalize_pretrained_model(pretrained_model):
    """
    References:
        `_get_checkpoint_filename` in python/training/checkpoint_utils.py
    """
    # pretrained_model maybe None
    if pretrained_model is not None:
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
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in var_list])
    uninitialized_vars = [v for (v, f) in zip(var_list, is_not_initialized) if not f]
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
    
def filter_variables(var_names_or_func, vars=None):
    if var_names_or_func is None:
        return vars
        
    if vars is None:
        vars = tf.trainable_variables()
    if hasattr(var_names_or_func, '__call__'):
        # the return value type could be variable name list or tf.Variable list.
        var_list = var_names_or_func(vars)
        var_names = []
        for var_item in var_list:
            if isinstance(var_item, (str, unicode)):
                var_names.append(var_item)
            elif isinstance(var_item, tf.Variable):
                var_names.append(var_item.op.name)
        var_list = [var for var in vars if var.op.name in var_names]
    elif hasattr(var_names_or_func, '__iter__'):
        var_list = [var for var in vars if var.op.name in var_names_or_func]
    else:
        raise ValueError("var_names_or_func should be sequence, "
                        "function with one argument (list of tf.Variable). "
                        "Got {}.".format(type(var_names_or_func)))
    return var_list
    
def build_variable_average_op(decay, num_updates=None, var_list=None, zero_debias=False):
    """Build variable moving average operation.

    Args:
        num_updates: Optional count of number of updates applied to variables.
        zero_debias: If `True`, zero debias moving-averages that are initialized with tensors.
        var_list: A list of Variable objects or Variable's name.
        
    Returns:
        An Operation that updates the moving averages.
    """
    def check_average_variable(tensor_to_check, var_list):
        for var in var_list:
            if isinstance(var, (str, unicode)):
                if var.split(':')[0] == tensor_to_check.op.name:
                    return True
            elif isinstance(var, tf.Variable):
                if var.name == tensor_to_check.name:
                    return True
            else:
                raise ValueError('Invalid type. Got {}.'.format(type(var)))
        return False

    variable_averages = tf.train.ExponentialMovingAverage(decay, num_updates, zero_debias)
    var_list = filter_variables(var_list, tf.trainable_variables())
    if var_list is None:
        variables_to_average = tf.trainable_variables()
    else:
        variables_to_average = [each for each in tf.trainable_variables()
                                if check_average_variable(each, var_list)]
    variables_averages_op = variable_averages.apply(variables_to_average)
    
    return variables_averages_op
    
def get_checkpoint_vars(filename, var_names=None, only_shape=False, sort=False):
    """Returns an OrderedDict of params indexed by name from the checkpoint.

    References:
        `tf.train.list_variables` or `tf.contrib.framework.list_variables`.
        caffe.Net.params in PyCaffe
    """
    if tf.gfile.IsDirectory(filename):
        filename = tf.train.latest_checkpoint(filename)
    if filename is None:
        raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                     "given directory %s" % filename)
    reader = tf.train.NewCheckpointReader(filename)
    
    variable_map = reader.get_variable_to_shape_map()
    if var_names is None:
        var_names = variable_map.keys()
    if sort:
        var_names = sorted(var_names)
        
    vars_dict = OrderedDict()
    if only_shape:
        for var_name in var_names:
            vars_dict[var_name] = variable_map[var_name]
    else:
        for var_name in var_names:
            vars_dict[var_name] = reader.get_tensor(var_name)
    return vars_dict
    
def multiply_gradients(grads_and_vars, gradient_multipliers):
    """Multiply specified gradients.

    Args:
        grads_and_vars: A list of gradient to variable pairs (tuples).
        gradient_multipliers: A map from either `Variables` or `Variable` op names
            to the coefficient by which the associated gradient should be scaled.

    Returns:
        The updated list of gradient to variable pairs.

    Raises:
        ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers`
        is empty or None or if `gradient_multipliers` is not a dictionary.
        
    References:
        tensorflow/contrib/slim/python/slim/learning.py
    """
    if not isinstance(grads_and_vars, list):
        raise ValueError('`grads_and_vars` must be a list.')
    if not gradient_multipliers:
        raise ValueError('`gradient_multipliers` is empty.')
    if not isinstance(gradient_multipliers, dict):
        raise ValueError('`gradient_multipliers` must be a dict.')

    multiplied_grads_and_vars = []
    for grad, var in grads_and_vars:
        if var in gradient_multipliers or var.op.name in gradient_multipliers:
            key = var if var in gradient_multipliers else var.op.name
            if grad is None:
                raise ValueError('Requested multiple of `None` gradient.')

            multiplier = gradient_multipliers[key]
            if not isinstance(multiplier, tf.Tensor):
                multiplier = tf.constant(multiplier, dtype=grad.dtype)

            if isinstance(grad, tf.IndexedSlices):
                tmp = grad.values * multiplier
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad *= multiplier
        multiplied_grads_and_vars.append((grad, var))
    return multiplied_grads_and_vars

def get_or_create_global_step(graph=None):
    """Returns and create (if necessary) the global step variable.

    Args:
        graph: The graph in which to create the global step. If missing, use default
            graph.

    Returns:
        the tensor representing the global step variable.
    
    References:
        `tf.train.get_global_step`
        `tf.train.create_global_step` or `tf.contrib.framework.create_global_step`
        `tf.train.get_or_create_global_step` or `tf.contrib.framework.get_or_create_global_step`
    """
    return tf.contrib.framework.get_or_create_global_step(graph)
    
def convert_dict_value_to_tensor(ordered_dict):
    converted_dict = OrderedDict()
    for key, value in ordered_dict.items():
        converted_dict[key] = tf.convert_to_tensor(value)
    return converted_dict
    
def get_name_scope():
    try:
        return tf.get_default_graph().get_name_scope()
    except:
        return tf.get_default_graph()._name_stack
        
def normalize_tensor_name(name):
    if name.find(':') == -1:
        return name + ':0'
    else:
        return name
        
def get_tensor_name_in_scope(tensor_name, scope=None):
    tensor_name = normalize_tensor_name(tensor_name)
    if (scope is None) or (scope == ''):
        return tensor_name
    else:
        scope = scope.rstrip('/')
        return '{}/{}'.format(scope, tensor_name)
        
def get_tensor_by_name(tensor_name, scope=None, graph=None):
    if graph is None:
        graph = tf.get_default_graph()
    tensor_name = get_tensor_name_in_scope(tensor_name, scope)
    return graph.get_tensor_by_name(tensor_name)
    
    