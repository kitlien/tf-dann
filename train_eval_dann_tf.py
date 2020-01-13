#matplotlib inline

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import datasets
import importlib
import logging
import time
from datetime import datetime
from tensorflow.python import pywrap_tensorflow
from common import dap_func_wrapper
import _init_paths
import diluface.utils as utils
from flip_gradient import flip_gradient


tf.logging.set_verbosity(tf.logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


flags = tf.app.flags
flags.DEFINE_string('checkpoint_path', './trained_models/source_train_20190603_225957_finetune_newid/model_iter_58000.ckpt',
                    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_string('crop_type', '224x224_with_coord',
                    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_string('train_types', 'dann_all_train', 'train types')
flags.DEFINE_string('file_names', 'datagate/source_target_pair_0603_all.txt',
                    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_integer('batch_size', 48, 'Batch size.')
flags.DEFINE_integer('epoch_size', 10, 'Number of batches per epoch.')
flags.DEFINE_integer('save_model_steps', 1000, 'save steps.')
flags.DEFINE_integer('max_nrof_epochs',5,'epoch size.')
flags.DEFINE_integer('base_lr',0.0001,'base learning rate.')
flags.DEFINE_integer('image_height',224,'image size.')
flags.DEFINE_integer('image_width',224,'image size.')

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
#dataset.next_batch_source()

print (dataset._num_class_source)
print (dataset._num_class_target)

record_defaults = [[''], [0], [0], [''], [0], [0]]

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
            #print (len(restore_variables_dict),var_name)
           
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
            if var.name.startswith('Logits'):
                #var_name = var_name.replace('Logits', 'label_predictor/Logits', 1)
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

def build_model_source(source_image_batch, target_image_batch,\
            source_domain_label_batch, target_domain_label_batch, \
            source_labels_batch, target_labels_batch, learning_rate_placeholder, \
            dann_adapt_placeholder, batch_size_placeholder, global_step):
    embedding_size = 128
    scale = 4.5
    weight_decay = 0.0005
    num_classes = dataset._num_class_source
    model_def = 'models.ShuffleNet_v2'
    network = importlib.import_module(model_def)
    
    image_batch = tf.concat([source_image_batch, target_image_batch], 0)
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
    #with tf.variable_scope('label_predictor'):
    source_features = tf.slice(prelogits, [0, 0], [FLAGS.batch_size, -1])
    
    classify_feats = source_features
    #classify_feats = prelogits
    classify_labels = source_labels_batch
    
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
    
    output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Logits')
    regular_loss = tf.add_n([pred_loss] + regularization_losses, name='regular_loss')
    regular_train_op = tf.train.RMSPropOptimizer(learning_rate_placeholder, decay=0.9, \
        momentum=0.9, epsilon=1.0).minimize(regular_loss, global_step = global_step)
        
    with tf.variable_scope('domain_predictor'):
        
        domain_label = tf.concat([source_domain_label_batch, target_domain_label_batch], 0)
        domain_label_onehot = tf.one_hot(domain_label, 2)
        # Flip the gradient when backpropagating through this operation
        feat = flip_gradient(prelogits, dann_adapt_placeholder)
        
        d_h_fc0 = slim.fully_connected(feat, 1024, activation_fn = tf.nn.relu, scope = 'fc1')
        d_h_fc1 = slim.fully_connected(d_h_fc0, 1024, activation_fn = tf.nn.relu, scope = 'fc2')
        d_logits = slim.fully_connected(d_h_fc1, 2, activation_fn = None, scope = 'fc3')
        
        domain_pred = tf.nn.softmax(d_logits)
        domain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d_logits, labels=domain_label))

        dann_loss = pred_loss + domain_loss
        dann_loss = tf.add_n([dann_loss] + regularization_losses, name='dann_loss')
        dann_train_op = tf.train.RMSPropOptimizer(learning_rate_placeholder, decay=0.9, \
            momentum=0.9, epsilon=1.0).minimize(dann_loss, global_step = global_step)
            
        correct_domain_pred = tf.equal(tf.argmax(domain_label_onehot, 1), tf.argmax(domain_pred, 1))
        domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))
    
    return dann_train_op, dann_loss, pred_loss, domain_acc, scale_inner
    
       
def _learning_rate_fn(current_learning_rate, steps):
    if steps != 0 and steps % 100000 == 0:
        return current_learning_rate * 0.98
    return current_learning_rate
    
def train_and_evaluate(training_mode, graph, model, logdir, num_steps=8600,  verbose=False):
    """Helper to run the model with different training modes."""
    
    log_filename = os.path.join(logdir,'train_log.txt')
    set_logger(log_filename, logging.INFO)
    
    logging.info('start training')
    old_time = time.time()
    
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        filenames_placeholder = tf.placeholder(tf.string, name='image_paths')
        learning_rate_placeholder = tf.placeholder(tf.float32, [])
        dann_adapt_placeholder = tf.placeholder(tf.float32, [])
        
        input_queue = tf.FIFOQueue(capacity = 32, dtypes=tf.string, shapes=[()])
        enqueue_op = input_queue.enqueue_many(filenames_placeholder) 
        
        dap_func = dap_func_wrapper(FLAGS.image_width, FLAGS.image_height)
        reader = tf.TextLineReader()
        images_and_labels_list = []
        
        num_dap_threads = 4
        for _ in range(num_dap_threads):
            key, record = reader.read(input_queue)
            decoded = tf.decode_csv(record, record_defaults=record_defaults, field_delim=',')
            source_content = tf.read_file(decoded[0])
            source_image = dap_func(source_content, augment = True)
            source_class_label = decoded[1]
            source_domain_label = decoded[2]
            target_content = tf.read_file(decoded[3])
            target_class_label = decoded[4]
            target_domain_label = decoded[5]
            target_image = dap_func(target_content, augment = True)
            images_and_labels_list.append([record, source_image, target_image, \
                    source_class_label, target_class_label, source_domain_label, \
                    target_domain_label])
        
        record_batch, source_image_batch, target_image_batch, source_labels_batch, target_labels_batch, \
            source_domain_label_batch, target_domain_label_batch = tf.train.batch_join(
            images_and_labels_list, batch_size=batch_size_placeholder, 
            enqueue_many = False,
            capacity=2 * num_dap_threads * FLAGS.batch_size,
            allow_smaller_final_batch=True)
            
        dann_train_op, dann_loss, pred_loss, domain_acc, scale_inner = \
            build_model_source(source_image_batch, target_image_batch,\
            source_domain_label_batch, target_domain_label_batch, source_labels_batch, \
            target_labels_batch, learning_rate_placeholder,dann_adapt_placeholder, \
            batch_size_placeholder, global_step)
        
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
            
            filenames = [FLAGS.file_names]
            i_step = 0
            for epoch in range(1, FLAGS.max_nrof_epochs):
                # Enqueue one epoch of image paths and labels
                num_examples = sum([utils.get_file_line_count(name) for name in filenames])
                num_batches = (num_examples + FLAGS.batch_size - 1) // FLAGS.batch_size
                step = sess.run(global_step, feed_dict=None)
                sess.run(enqueue_op, {filenames_placeholder: filenames})
                
                for i in range(num_batches):
                    
                    p = float(i_step) / (FLAGS.max_nrof_epochs * num_batches)
                    l = 2. / (1. + np.exp(-10. * p)) - 1
                    i_step = i_step + 1
                    
                    batch_size_actual = min(num_examples - i * FLAGS.batch_size, FLAGS.batch_size)
                    source_lr = _learning_rate_fn(FLAGS.base_lr, step)
                    feed_dict = {learning_rate_placeholder: source_lr, \
                                     batch_size_placeholder: batch_size_actual, \
                                     dann_adapt_placeholder: l}
                    start_time = time.time()
                    _, batch_loss, ploss, d_acc, scale, step = sess.run([dann_train_op, \
                        dann_loss, pred_loss, domain_acc, scale_inner, global_step], feed_dict = feed_dict)
 
                    elapsed=time.time()-start_time
                    logging.info(('epoch:{} step:{} iter:{}/{} Time: {:.4f}s  loss: {:.4f} ploss: {:.4f}  d_acc: {:.4f}  lr: {:.4f}  scale:{:.4f}  l:{:.6f}'\
                        .format(epoch, step, i, num_batches, \
                        elapsed, batch_loss, ploss, d_acc, source_lr, scale, l)))
                    
                    if step % FLAGS.save_model_steps == 0:
                        filename = os.path.join(logdir, 'model_iter_{:d}'.format(step) + '.ckpt')
                        saver.save(sess, filename)

time_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
sub_save_dir = os.path.join(os.path.expanduser(FLAGS.save_dir), FLAGS.train_types + '_' + time_str)
base_name_pre = os.path.basename(sub_save_dir)
print (base_name_pre)
makedirs_p(sub_save_dir)

print('\nSource only training')
graph_s = 'placeholder'
model_s = 'placeholder'
train_and_evaluate('source', graph_s, model_s, sub_save_dir, 100000)