from common_online import dap_func_wrapper
import tensorflow as tf
import time
import os
import numpy as np
import csv
import cv2
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

crop_type = '224x224_with_coord'
dap_func = dap_func_wrapper(crop_type)

filename = './headshot_landmarks/train_list_new_module.txt'
log_dir = '/storage-server1/workspace/lianjie/work/FaceGeneralize/image_undistort/datasets/log_dir_new_module'
oridir = '/storage-server1/workspace/lianjie/work/FaceGeneralize/image_undistort/datasets/photo_new_module_headshot'

num_workers = 8

if not os.path.isfile(filename):
    file_new = open(filename, 'w')
    num_workers = 8
    path_vector = []
    for i in range(num_workers):
        result_filename = os.path.join(log_dir, 'task_{}_file_list_result.txt'.format(i))
        path_vector.append(result_filename)
    print (len(path_vector))
    for ip in range(len(path_vector)):
        path_list = open(path_vector[ip]).readlines()
        for line in path_list:
            values = line.strip().split(' ')
            filepath = os.path.join(oridir, values[-1])
            file_new.write(filepath)
            file_new.write(' ')
            for i in range(len(values)-1):
                file_new.write(values[i])
                file_new.write(' ')
            file_new.write('\n')
    file_new.close()

#file_name_string = './headshot_landmarks/train_list_raw.csv'
file_name_string = './headshot_landmarks/train_list_2w_widcard.txt'

filename_queue = tf.train.string_input_producer([filename])

record_defaults = [[''], [0], [0]] + [[0.0]] * 10 + [['']]

reader = tf.TextLineReader()
num_dap_threads = 4
images_and_labels_list = []
for _ in range(num_dap_threads):
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=record_defaults, field_delim=' ')
    image, filename = dap_func(decoded, augment = False)
    images_and_labels_list.append([image, filename])
    
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    ii = 0
    while True:
        cur_image, cur_filename = sess.run([image, filename])
        print (cur_filename)
        
        cv2.imwrite(os.path.join('./headshot_landmarks/undistort_results', os.path.basename(cur_filename)),cur_image)