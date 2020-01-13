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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


flags = tf.app.flags
flags.DEFINE_string('checkpoint_path', './trained_models/source_train_20190427_215228/model_iter_26000.ckpt',
                    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
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
        
class shufflenet_v2(object):
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        self.num_classes = dataset._num_class_source
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.int32, [None])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.train = tf.placeholder(tf.bool, [])
        self.l = tf.placeholder(tf.float32, [])
        self.num_images = dataset._num_train_source_batch
        self.label_onehot_y = tf.one_hot(self.y, self.num_classes)
        
        self.embedding_size = 128
        self.scale = 4.5
        self.weight_decay = 0.0005
        
        model_def = 'models.ShuffleNet_v2'
        network = importlib.import_module(model_def)
        self.prelogits, endpoints = network.inference(self.X,
                                    embedding_size = self.embedding_size,
                                    keep_probability = 0.6,
                                    weight_decay = self.weight_decay,
                                    use_batch_norm = True,
                                    phase_train = True,
                                    scope = None)
        
        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
        
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            # all_features = lambda: self.prelogits
            # source_features = lambda: tf.slice(self.prelogits, [0, 0], [FLAGS.batch_size, -1])
            # classify_feats = tf.cond(self.train, source_features, all_features)
            
            classify_feats = self.prelogits
            
            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0], [FLAGS.batch_size])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)
            
            self.scale_inner = tf.get_variable('Logits/scale', (), dtype=classify_feats.dtype,
                        initializer=tf.constant_initializer(self.scale), 
                        regularizer=slim.l2_regularizer(self.weight_decay))
                                        
            self.logits = normalized_linear_layer(classify_feats, self.num_classes, 
                            weights_initializer=tf.truncated_normal_initializer(stddev=1e-1),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            scale_inputs=self.scale_inner,
                            normalize_weights=False,
                            name='Logits')
            
            self.pred = tf.nn.softmax(self.logits)
            source_logits = tf.slice(self.logits, [0, 0], [FLAGS.batch_size, -1])
            self.pred_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.classify_labels, logits=source_logits), name='l2_softmax_loss')
                
        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            d_W_fc0 = weight_variable([self.embedding_size, 5])
            d_b_fc0 = bias_variable([5])
            d_h_fc0 = tf.nn.relu(tf.matmul(self.prelogits, d_W_fc0) + d_b_fc0)
                
            feature_ex = tf.expand_dims(d_h_fc0, 2)
            softmax_out_ex = tf.expand_dims(self.pred, 1)
            outer_product_out = tf.matmul(feature_ex, softmax_out_ex)
            
            class_weight = tf.reduce_mean(self.pred, axis = 0)
            class_weight = (class_weight / tf.reduce_max(class_weight))
            
            self.domain_loss = tf.constant(0.0)
            for i in range(dataset._num_class_target):
                print (i)
                # Flip the gradient when back-propagating through this operation
                cur_l = self.l * class_weight[i]
                feat = flip_gradient(outer_product_out[:,:,i], cur_l)
                #feat = flip_gradient(self.prelogits, self.l)
                d_W_fc1 = weight_variable([5, 2])
                d_b_fc1 = bias_variable([2])
                d_logits = tf.nn.relu(tf.matmul(feat, d_W_fc1) + d_b_fc1)
                
                self.domain_pred = tf.nn.softmax(d_logits)
                self.domain_loss = tf.add(self.domain_loss, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)))
                
graph_s = tf.get_default_graph()
with graph_s.as_default():
    model_s = shufflenet_v2()
    learning_rate = tf.placeholder(tf.float32, [])
    #source_label_onehot = tf.slice(model_s.label_onehot_y, [0, 0], [FLAGS.batch_size, -1])
    label_one_hot = model_s.label_onehot_y
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    pred_loss = model_s.pred_loss
    domain_loss = model_s.domain_loss
    dann_loss = pred_loss + domain_loss
    
    regular_loss = tf.add_n([pred_loss] + regularization_losses, name='regular_loss')
    dann_loss = tf.add_n([dann_loss] + regularization_losses, name='dann_loss')
    #tf.summary.scalar('dann_loss',dann_loss)
    
    #regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
    regular_train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0).minimize(regular_loss)
    dann_train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0).minimize(dann_loss)
    scale_inner = model_s.scale_inner
    
    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(label_one_hot, 1), tf.argmax(model_s.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model_s.domain, 1), tf.argmax(model_s.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))
       
def _learning_rate_fn(current_learning_rate, steps):
    if steps != 0 and steps % 100000 == 0:
        return current_learning_rate * 0.1
    return current_learning_rate

def train_and_evaluate(training_mode, graph, model,logdir, num_steps=8600,  verbose=False):
    """Helper to run the model with different training modes."""
    
    log_filename = os.path.join(logdir,'train_log.txt')
    set_logger(log_filename, logging.INFO)
    
    logging.info('start training')
    
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        default_to_restore = tf.trainable_variables()
        #default_to_restore += tf.get_collection(tf.GraphKeys.RESTORE_VARIABLES)
        saver = tf.train.Saver(default_to_restore, max_to_keep=5)
        if FLAGS.checkpoint_path:
            _model_restore_fn(
                FLAGS.checkpoint_path, sess, default_to_restore, FLAGS.restore_from_base_network)
        
        #merged = tf.summary.merge_all() 
        writer = tf.summary.FileWriter('./tensorboard_model', sess.graph)
        # Training loop
        for i in range(num_steps):
            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01 / (1. + 10 * p)**0.75

            source_lr = _learning_rate_fn(0.0001, i)
            # Training step
            if training_mode == 'dann':
                source_name, X0, y0, batch_source_size = dataset.next_batch_source()
                target_name, X1, y1, batch_target_size = dataset.next_batch_target()
                
                X = np.vstack([X0, X1])
                y = np.hstack([y0, y1])
                
                domain_labels = np.vstack([np.tile([1., 0.], [batch_source_size, 1]),
                           np.tile([0., 1.], [batch_target_size, 1])])
                
                _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                    [dann_train_op, dann_loss, domain_loss, pred_loss, domain_acc, label_acc],
                    feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                               model.train: True, model.l: l, learning_rate: source_lr})
                if i % 10 == 0:
                    logging.info('iter:{} loss: {:.4} ploss: {:.4} d_acc: {:.4}  p_acc: {:.4}  p: {:.4}  l: {:.4}  lr: {:.4}'.format(i,
                            batch_loss, ploss, d_acc, p_acc, p, l, source_lr))
                    #writer.add_summary(merge,i) 
                if i == 0 or (i + 1) % 1000 == 0:
                    filename = os.path.join(logdir, 'dann_model_iter_{:d}'.format(i+1) + '.ckpt')
                    saver.save(sess, filename)
            elif training_mode == 'source':
                _, X, y, batch_source_size = dataset.next_batch_source()
                _, batch_loss, ploss, p_acc, scale = sess.run([regular_train_op, regular_loss, pred_loss, label_acc, scale_inner],
                                     feed_dict={model.X: X, model.y: y, model.train: False,
                                                model.l: l, learning_rate: source_lr})
                if i % 10 == 0:
                    logging.info('iter:{}/{} loss: {:.4} ploss: {:.4}  p_acc: {:.4}  p: {:.4}  l: {:.4}  lr: {:.4}  scale:{}'.format(i,
                            model.num_images, batch_loss, ploss, p_acc, p, l, source_lr, scale))
                if i == 0 or (i + 1) % 1000 == 0:
                    filename = os.path.join(logdir, 'model_iter_{:d}'.format(i+1) + '.ckpt')
                    saver.save(sess, filename)
        #writer.close()
time_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
sub_save_dir = os.path.join(os.path.expanduser(FLAGS.save_dir), 'san_train' + '_' + time_str)
base_name_pre = os.path.basename(sub_save_dir)
print (base_name_pre)
makedirs_p(sub_save_dir)

print('\nDomain adaptation training')
train_and_evaluate('dann', graph_s, model_s, sub_save_dir, 100000)

#print('\nSource only training')
#train_and_evaluate('source', graph_s, model_s, sub_save_dir, 100000)
