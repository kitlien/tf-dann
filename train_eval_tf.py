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
import time
from datetime import datetime
from tensorflow.python import pywrap_tensorflow
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


flags = tf.app.flags
flags.DEFINE_string('checkpoint_path', './trained_models/source_train_20190516_215124/model_iter_41.ckpt',
                    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_integer('batch_size', 48, 'Batch size.')
flags.DEFINE_integer('epoch_size', 1000, 'Number of batches per epoch.')
flags.DEFINE_integer('max_nrof_epochs',10000,'epoch size.')
flags.DEFINE_boolean('restore_from_base_network', False,
                     'Restore model only for base network.')
flags.DEFINE_string('save_dir', './trained_models/',
                    'The path to save a checkpoint and model file.')
                    
FLAGS = flags.FLAGS

dataset = datasets.datagate_dataset(is_training = True, 
                               batch_size = FLAGS.batch_size, 
                               input_height= 224, 
                               input_width= 224, 
                               input_channels= 3)
print (dataset._num_class_source)
print (dataset._num_class_target)
def makedirs_p(path, mode=0o755):
    """Similar to `mkdir -p`
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    elif not os.path.isdir(path):
        raise IOError("%r exists but is not a directory" % path)
        
def set_logger(filename, level=logging.INFO):
    logging.basicConfig(level=level,
                    format='%(message)s',
                    datefmt='%Y-%M-%d %H:%M:%S',
                    filename=filename,
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(console)
    
def _model_restore_fn(ckpt_path, sess, default_to_restore, restore_from_base_network=False):
    if restore_from_base_network:
        tf.logging.info('Load pretrained model in {}'.format(ckpt_path))
        model_variables = slim.get_model_variables()
        def get_variable_name(var):
            return var.op.name
        restore_variables_dict = {}
        for var in model_variables:
            var_name = get_variable_name(var)
            restore_variables_dict[var_name] = var
            print (len(restore_variables_dict),var_name)
           
        restorer = tf.train.Saver(restore_variables_dict)
        restorer.restore(sess, ckpt_path)
    else:
        def get_variable_name(var):
            return var.op.name
        restore_variables_dict = {}
        for var in default_to_restore:
            var_name = get_variable_name(var)
            print (var_name)
            if var.name.startswith('ShuffleNet_v2'):
                restore_variables_dict[var_name] = var
            if var.name.startswith('label_predictor/Logits'):
                #var_name = var_name.replace('label_predictor/Logits', 'Logits', 1)
                restore_variables_dict[var_name] = var
        print (len(restore_variables_dict))
        tf.logging.info('Load pretrained model in {}'.format(ckpt_path))
        restorer = tf.train.Saver(var_list=restore_variables_dict)
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

def build_model_source(image_batch, labels_batch, learning_rate_placeholder):
    embedding_size = 128
    scale = 4.5
    weight_decay = 0.0005
    num_classes = dataset._num_class_source
    model_def = 'models.ShuffleNet_v2'
    network = importlib.import_module(model_def)
    prelogits, endpoints = network.inference(image_batch,
                                embedding_size = embedding_size,
                                keep_probability = 0.6,
                                weight_decay = weight_decay,
                                use_batch_norm = True,
                                phase_train = True,
                                scope = None)
    
    # Switches to route target examples (second half of batch) differently
    # depending on train or test mode.
    # MLP for class prediction
    with tf.variable_scope('label_predictor'):
        classify_feats = prelogits
        classify_labels = labels_batch
        
        scale_inner = tf.get_variable('Logits/scale', (), dtype=classify_feats.dtype,
                    initializer=tf.constant_initializer(scale), 
                    regularizer=slim.l2_regularizer(weight_decay))
                                    
        logits = normalized_linear_layer(classify_feats, num_classes, 
                        weights_initializer=tf.truncated_normal_initializer(stddev=1e-1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        scale_inputs=scale_inner,
                        normalize_weights=False,
                        name='Logits')
                                
    pred_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=classify_labels, logits=logits), name='l2_softmax_loss')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regular_loss = tf.add_n([pred_loss] + regularization_losses, name='regular_loss')
    regular_train_op = tf.train.RMSPropOptimizer(learning_rate_placeholder, decay=0.9, momentum=0.9, epsilon=1.0).minimize(regular_loss)
    
    return regular_train_op, regular_loss, pred_loss, scale_inner
    
       
def _learning_rate_fn(current_learning_rate, steps):
    if steps != 0 and steps % 100000 == 0:
        return current_learning_rate * 0.1
    return current_learning_rate
    
def train_and_evaluate(training_mode, graph, model, logdir, num_steps=8600,  verbose=False):
    """Helper to run the model with different training modes."""
    
    log_filename = os.path.join(logdir,'train_log.txt')
    set_logger(log_filename, logging.INFO)
    
    logging.info('start training')
    old_time = time.time()
    
    with tf.Graph().as_default():
        
        image_list, label_list = dataset._train_source_images, dataset._train_source_labels
        
        labels = tf.convert_to_tensor(np.array(label_list), dtype=tf.int64)
        range_size = tf.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                 shuffle=True, seed=None, capacity=32)
            
        index_dequeue_op = index_queue.dequeue_many(FLAGS.batch_size*FLAGS.epoch_size, 'index_dequeue')
        
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
        learning_rate_placeholder = tf.placeholder(tf.float32, [])
        
        input_queue = tf.FIFOQueue(capacity = 10000000, dtypes=[tf.string, tf.int64], shapes=[(1,), (1,)])
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        
        images_and_labels_list = []
        reader = tf.TextLineReader()
        num_dap_threads = 4
        for _ in range(num_dap_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, 3)
                image = tf.image.random_flip_left_right(image)
                image = tf.cast(image, tf.float32)
                # preprocess
                image -= 127.5
                image /= 128.0
                # set shape, required
                image.set_shape([224, 224, 3])
                images.append(image)
            images_and_labels_list.append([images, label])
        
        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels_list, batch_size=batch_size_placeholder, 
            enqueue_many=True,
            capacity=4 * num_dap_threads * FLAGS.batch_size,
            allow_smaller_final_batch=True)
            
        regular_train_op, regular_loss, pred_loss, scale_inner = \
            build_model_source(image_batch, labels_batch, learning_rate_placeholder)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        
        with sess.as_default():
            tf.global_variables_initializer().run()
            default_to_restore = tf.trainable_variables()
            #default_to_restore += tf.get_collection(tf.GraphKeys.RESTORE_VARIABLES)
            saver = tf.train.Saver(default_to_restore, max_to_keep=5)
            if FLAGS.checkpoint_path:
                _model_restore_fn(
                    FLAGS.checkpoint_path, sess, default_to_restore, FLAGS.restore_from_base_network)
            
            for epoch in range(1, FLAGS.max_nrof_epochs):
                index_epoch = sess.run(index_dequeue_op)
                label_epoch = np.array(label_list)[index_epoch]
                image_epoch = np.array(image_list)[index_epoch]
                
                # Enqueue one epoch of image paths and labels
                labels_array = np.expand_dims(np.array(label_epoch),1)
                image_paths_array = np.expand_dims(np.array(image_epoch),1)
                sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
                
                for i in range(FLAGS.epoch_size):
                    source_lr = _learning_rate_fn(0.01, i)
                    start_time = time.time()
                    _, batch_loss, ploss, scale = sess.run([regular_train_op, regular_loss, pred_loss, scale_inner],
                                         feed_dict={learning_rate_placeholder: source_lr, \
                                         batch_size_placeholder: FLAGS.batch_size})
 
                    elapsed=time.time()-start_time
                    logging.info(('iter:{}  epoch:{}/{} Time: {:.4f}s loss: {:.4f} ploss: {:.4f}  lr: {:.4f}  scale:{:.4f}'.format(i, \
                    epoch, dataset._num_train_source_batch, elapsed, batch_loss, ploss, source_lr, scale)))
                
                filename = os.path.join(logdir, 'model_iter_{:d}'.format(epoch) + '.ckpt')
                saver.save(sess, filename)

time_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
sub_save_dir = os.path.join(os.path.expanduser(FLAGS.save_dir), 'source_train' + '_' + time_str)
base_name_pre = os.path.basename(sub_save_dir)
print (base_name_pre)
makedirs_p(sub_save_dir)

# print('\nDomain adaptation training')
# train_and_evaluate('dann', graph_s, model_s, sub_save_dir, 100000)

print('\nSource only training')
graph_s = 'hh'
model_s = 'hh'
train_and_evaluate('source', graph_s, model_s, sub_save_dir, 100000)