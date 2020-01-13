import os
import sys
import numpy as np
from scipy import misc
import tensorflow as tf
from scipy import misc
import utils_tf
import time
from collections import OrderedDict
import logging
from scipy import interpolate
from sklearn import metrics

import _init_paths
import diluface.utils as utils
from datetime import datetime

def dap_func_wrapper(image_height, image_width):
    def dap_func(path, augment=False):
        image = misc.imread(path, mode=None)
        image = misc.imresize(image, (image_height, image_width))
        image = image.astype(np.float32)
        image -= 127.5
        image /= 128.0
        return image
    return dap_func
    
def load_images(image_paths, preprocess_func=None):
    images = []
    preprocess_func = preprocess_func or (lambda x: x)
    for path in image_paths:
        images.append(preprocess_func(path))
    return images

def normalize_image_shape(image_shape):
    image_shape = list(image_shape)
    assert len(image_shape) in (2, 3)
    if len(image_shape) == 2:
        image_shape += [1]
    return image_shape
    
def build_network_tf(model_def, model_weight, image_shape, embedding_size=None, 
                    use_batch_norm=False, dtype=tf.float32):
    tf.reset_default_graph()
    h, w, c = normalize_image_shape(image_shape)
    inputs = tf.placeholder(dtype, shape=(None, h, w, c), name='inputs')
    embeddings, _ = utils_tf.build_inference(model_def, inputs, 
                                            embedding_size=embedding_size, 
                                            keep_probability=1.0, 
                                            weight_decay=0.0, 
                                            use_batch_norm=use_batch_norm, 
                                            phase_train=False)
    outputs = tf.identity(embeddings, 'outputs')
    vars_to_restore = tf.trainable_variables()
    sess = tf.Session(config=utils_tf.get_default_sess_config())
    utils_tf.restore(sess, model_weight, vars_to_restore, model_restore_func=None)
    return sess, inputs, outputs

def get_features_tf(sess, inputs, outputs, image_paths, batch_size, preprocess_func, 
                    use_normalized=True, num_steps=None):
    print("get_features_tf")
    embedding_size = outputs.get_shape()[1]

    num_images = len(image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size
    feature_array = np.zeros((num_images, embedding_size))
    
    for i in range(num_batches):
        start = time.time()
        start_ind = i * batch_size
        end_ind = min((i + 1) * batch_size, num_images)
        paths_batch = image_paths[start_ind:end_ind]
        
        images = load_images(paths_batch, preprocess_func=preprocess_func)
            
        feed_dict = { inputs: images}
        if num_steps is None:
            feature_array[start_ind:end_ind, :] = sess.run(outputs, feed_dict=feed_dict)
        else:
            output, summary = utils.benchmark(lambda: sess.run(outputs, feed_dict=feed_dict), num_steps)
            print('Across {} steps: '.format(num_steps))
            print('mean: {:.3f} stddev: {:.3f}, max: {:.3f}, min: {:.3f} (sec/step)'.format(
                summary['mean'], summary['stddev'], summary['max'], summary['min']))
            feature_array[start_ind:end_ind, :] = output
            #feature_array[paths_batch, :] = output
        end = time.time()
        print('[{}/{}] {} time:{:.4f}'.format(end_ind, num_images, os.path.basename(image_paths[end_ind - 1]), end-start))
    if use_normalized is True:
        feature_array /= np.linalg.norm(feature_array, axis=1, keepdims=True)
    return feature_array

def convert_feature_array_to_dict(key_list, feature_array):
    assert len(feature_array) == len(key_list)
    feature_dict = OrderedDict()
    for k, key in enumerate(key_list):
        feature_dict[key] = feature_array[k]
    return feature_dict

def evaluate_1vs1_global(probs, actual_issame, thresholds, interested_fars):
    """1:1 global evaluation
    """
    num_thresholds = len(thresholds)
    accs = np.zeros(num_thresholds)
    tprs = np.zeros(num_thresholds)
    fprs = np.zeros(num_thresholds)
    for idx, threshold in enumerate(thresholds):
        logging.debug('[{}/{}] Computing evalution metrics @ thresh={}'.format(idx+1, len(thresholds), threshold))
        predict_issame = np.greater(probs, threshold)
        tprs[idx], fprs[idx], accs[idx] = utils.calculate_tpr_fpr_acc(actual_issame, predict_issame)
        
    max_ind = np.argmax(accs)
    best_thresh = thresholds[max_ind]
    max_accuracy = accs[max_ind]
    logging.info('MaxAcc: {:.5f} @ Thresh={:.5f}'.format(max_accuracy, best_thresh))
    
    # Find the threshold that gives FAR = far_target
    f = interpolate.interp1d(fprs, thresholds, kind='slinear')
    for far_target in interested_fars:
        if np.max(fprs) >= far_target:
            try:
                threshold = f(far_target).item()
            except:
                threshold = 0.0
        else:
            threshold = 0.0
        predict_issame = np.greater(probs, threshold)
        tpr, fpr, _ = utils.calculate_tpr_fpr_acc(actual_issame, predict_issame)
        logging.info('TAR: {:.5f} @ FAR={:.9f}, Thresh={:.5f}'.format(tpr, fpr, threshold))
    logging.info('Area Under Curve (AUC): {:.7f}'.format(metrics.auc(fprs, tprs)))
    
    threshold = 0.5
    predict_issame = np.greater(probs, threshold)
    tpr, fpr, acc = utils.calculate_tpr_fpr_acc(actual_issame, predict_issame)
    logging.info('TAR: {:.5f} @ FAR={:.9f}, Thresh={:.5f}'.format(tpr, fpr, threshold))
    logging.info('ACC: {:.5f} @ Thresh={:.5f}'.format(acc, threshold))
    
    return tprs, fprs
    
def compute_1v1_tar_far(feature_dict, feature_dict_idcard):
    pos_pair_num, neg_pair_num = 0, 0
    pos_similarity, neg_similarity = [], []
    
    #feature_dict = np.load('datagate/live.npy').item()
    #feature_dict_idcard = np.load('datagate/idcard.npy').item()
    print (len(feature_dict), len(feature_dict_idcard))
    
    neg_i = 0
    live_i = 0
    for name in feature_dict:
        imgname = os.path.basename(name)
        idname = imgname.split('_')[0]
        live_i = live_i + 1
        if live_i%1000 == 0:
            #print(live_i, pos_pair_num, neg_pair_num)
            logging.info('live_i: {}, pos_pair_num: {}, neg_pair_num:{}'.format(live_i, pos_pair_num, neg_pair_num))
        for idcard_name in feature_dict_idcard:
            idcard_idname = os.path.basename(idcard_name).split('_')[0]
            #idcard_idname = os.path.basename(idcard_name).split('.')[0]
            if (idname == idcard_idname or (idname == 'lidy' and idcard_idname == 'zhangdy')):
                dist = np.linalg.norm(feature_dict[name] - feature_dict_idcard[idcard_name])
                similarity = 1 - dist*dist/4
                pos_similarity.append(similarity)
                #print (name, idcard_name, similarity)
                pos_pair_num = pos_pair_num + 1
            else:
                neg_i = neg_i + 1
                if neg_i == 100:
                    neg_pair_num = neg_pair_num + 1
                    dist = np.linalg.norm(feature_dict[name] - feature_dict_idcard[idcard_name])
                    similarity = 1 - dist*dist/4
                    neg_similarity.append(similarity)
                    #print (name, idcard_name, similarity)
                    neg_i = 0
    
    same_list = []
    for i in range(len(pos_similarity)):
        is_same = 1
        same_list.append(is_same)
    for i in range(len(neg_similarity)):
        is_same = 0
        same_list.append(is_same)
    prob_list = np.hstack((pos_similarity, neg_similarity))
    print (len(same_list),len(prob_list))
    
    thresholds = np.linspace(0, 1, 2000)
    interested_fars = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    tprs, fprs = evaluate_1vs1_global(prob_list, same_list, thresholds, interested_fars)
    
    # print ('pos pair:', pos_pair_num, 'neg_pair:', neg_pair_num)
    # alpha = [0.00001, 0.0001, 0.001 ,0.01, 0.1]
    # neg_similarity.sort(reverse = True)
    # for a in alpha:
        # threhold = neg_similarity[ int(a * len(neg_similarity)) ]
        # new_same = [i for i in pos_similarity if i > threhold]
        # str = 'TAR = %.4f @ FAR = %.5f, Thresh = %.4f '% \
            # (float(len(new_same)) / len(np.array(pos_similarity)), a, threhold)
        # print (str)
        
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

def get_image_path_hefei(live_dir, idcard_dir):
    live_list = []
    ii = 0
    for line in os.listdir(live_dir):
        live_list.append(os.path.join(live_dir, line))            
    idcard_list = []
    for line in os.listdir(idcard_dir):
        idcard_list.append(os.path.join(idcard_dir, line))
    return live_list, idcard_list
    
def get_image_path():
    live_file_list = open('./datagate/new_module_test_1vsn/new_live_photo_clean.txt').readlines()
    image_paths = []
    for line in live_file_list:
        image_paths.append(line.strip())
    
    idcard_file_list = open('./datagate/new_module_test_1vsn/new_idcard_photo_clean.txt').readlines()
    idcard_paths = []
    ii = 0
    for line in idcard_file_list:
            idcard_paths.append(line.strip())
    
    return image_paths, idcard_paths
if __name__ == '__main__':
    
    live_dir = '/storage-server1/workspace/lianjie/data/datagate_new_old_moudle_hefei/new_module_hefei_crop'
    idcard_dir = '/storage-server1/workspace/lianjie/data/datagate_new_old_moudle_hefei/idcardimage_hefei_crop'
    #image_paths, idcard_paths = get_image_path_hefei(live_dir, idcard_dir)
    image_paths, idcard_paths = get_image_path()
    tf_model_weight = '/storage-server1/workspace/lianjie/work/source/DiluFaceTrain/diluface_online_distort_crop/trained_models/224x224_rand_undistort_20190625_032230/224x224_rand_undistort_20190625_032230.ckpt-417000'
    #tf_model_weight = 'trained_models/source_train_20190603_225957/model_iter_58000.ckpt'

    dirname = os.path.dirname(tf_model_weight)
    basename = os.path.basename(tf_model_weight)
    time_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    
    log_filename = os.path.join(dirname, basename + '_test_1v1_' + time_str + '.txt')
    set_logger(log_filename, logging.INFO)
    
    tf_model_def = 'models.ShuffleNet_v2'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf_preprocess_func = dap_func_wrapper(224, 224)
    sess, inputs, outputs = build_network_tf(tf_model_def, 
                                            tf_model_weight,
                                            image_shape=(224,224,3), 
                                            embedding_size=128, 
                                            use_batch_norm=True)
    with sess.as_default():
        tf_features_probe = get_features_tf(sess, inputs, outputs, 
                                    image_paths=image_paths, 
                                    batch_size=64, 
                                    preprocess_func=tf_preprocess_func,
                                    num_steps=None)
        feature_dict_live = convert_feature_array_to_dict(image_paths, tf_features_probe)
        #np.save('datagate/live.npy', feature_dict_live)
        
        tf_features_idcard = get_features_tf(sess, inputs, outputs, 
                                    image_paths=idcard_paths, 
                                    batch_size=64, 
                                    preprocess_func=tf_preprocess_func,
                                    num_steps=None)
        feature_dict_idcard = convert_feature_array_to_dict(idcard_paths, tf_features_idcard)
        #np.save('datagate/idcard.npy', feature_dict_idcard)
        #feature_dict_live, feature_dict_idcard = [],[]
        compute_1v1_tar_far(feature_dict_live, feature_dict_idcard)
