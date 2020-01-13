# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import cv2
import os
import tensorflow as tf
import time

#train_list_old = 'datagate/train_list_source_2k.txt'
train_list_old = 'datagate/old_new_same_id_in_source.txt'
train_list_random = 'datagate/train_list_source_2k_random.txt'
val_list_old = 'datagate/val_list_old.txt'
train_list_new = 'datagate/old_new_same_id_in_target.txt'
val_list_new = 'datagate/val_list_new.txt'

def load_headshot_image(line):
    decode = line.split(' ')
    filename = decoded_record[0]
    tlx = decoded_record[1]
    tly = decoded_record[2]
    landmarks = decoded_record[3:13]
    image_cropped = align_and_crop_224x224_with_coord(image, landmarks, tlx, tly)
        
class datagate_dataset():
    def __init__(self, is_training, batch_size, image_type = 'raw',
                 input_height=256, input_width=256, input_channels=1):
        self._is_training = is_training
        self._batch_size = batch_size
        self._image_type = image_type
        self._num_train_source_batch = 0
        self._end_train_source_images = 0
        self._num_train_source_images = 0
        self._batch_train_source_iter = 0
        
        self._num_train_target_batch = 0
        self._end_train_target_images = 0
        self._num_train_target_images = 0
        self._batch_train_target_iter = 0
        self._source_class_num = 0
        self._num_class_source = 0
        self._num_class_target = 0
        
        self._train_source_images = []
        self._train_source_labels = []
        self._train_target_images = []
        self._train_target_labels = []
        self._input_h = input_height
        self._input_w = input_width
        self._input_c = input_channels
        
        class_dict = {}
        train_old_file = open(train_list_old)
        train_source_images = []
        train_source_labels = []
        for line in train_old_file.readlines():
            curLine = line.strip().split(',')
            self._train_source_images.append(curLine[0])
            self._train_source_labels.append(np.array(curLine[1],dtype = np.int32))
            class_dict[curLine[1]] = curLine[0]
        
        self._num_class_source = len(class_dict)
        
        val_old_file = open(val_list_old)
        val_source_images = []
        val_source_labels = []
        for line in val_old_file.readlines():
            curLine = line.strip().split(',')
            val_source_images.append(curLine[0])
            val_source_labels.append(np.array(curLine[1],dtype = np.float32))
        
        class_dict = {}
        train_new_file = open(train_list_new)
        train_target_images = []
        train_target_labels = []
        for line in train_new_file.readlines():
            curLine = line.strip().split(',')
            self._train_target_images.append(curLine[0])
            self._train_target_labels.append(np.array(curLine[1],dtype = np.int32))
            class_dict[curLine[1]] = curLine[0]
        
        self._num_class_target = len(class_dict)
        
        val_new_file = open(val_list_new)
        val_target_images = []
        val_target_labels = []
        for line in val_new_file.readlines():
            curLine = line.strip().split(',')
            val_target_images.append(curLine[0])
            val_target_labels.append(np.array(curLine[1],dtype = np.float32))
            
        self._num_train_source_images = len(self._train_source_images)
        self._num_train_source_batch = self._num_train_source_images // self._batch_size
        self._end_train_source_images = self._num_train_source_images % self._batch_size
        #if self._end_train_source_images > 0:
        #   self._num_train_source_batch += 1
           
        self._num_train_target_images = len(self._train_target_images)
        self._num_train_target_batch = self._num_train_target_images // self._batch_size
        self._end_train_target_images = self._num_train_target_images % self._batch_size
        if self._end_train_target_images > 0:
           self._num_train_target_batch += 1
           
    @property
    def num_train_source_images(self):
        return self._num_train_source_images
    @property
    def num_train_target_images(self):
        return self._num_train_target_images
    
    def num_train_source_iter(self):
        return self._batch_train_source_iter
        
    def num_train_target_iter(self):
        return self._batch_train_target_iter
        
    def next_batch_source(self):
        if self._is_training and self._batch_train_source_iter % self._num_train_source_batch == 0:
            randnum = random.randint(0, 6666)
            random.seed(randnum)
            random.shuffle(self._train_source_images)
            random.seed(randnum)
            random.shuffle(self._train_source_labels)
            self._batch_train_source_iter = 0
            
            # if not os.path.isfile(train_list_random):
                # file = open(train_list_random, 'w')
                # for i in range(len(self._train_source_images)):
                    # file.write(self._train_source_images[i])
                    # file.write(',')
                    # file.write(str(self._train_source_labels[i]))
                    # file.write('\n')
                    # #print (self._train_source_images[i], self._train_source_labels[i])
                # file.close()
                
        start_iter = self._batch_size * self._batch_train_source_iter
        end_iter = self._batch_size * (self._batch_train_source_iter + 1)
        if end_iter > self._num_train_source_images:
            end_iter = self._num_train_source_images
        inteval_iter = end_iter - start_iter
        self._batch_train_source_iter += 1
        batch_bunch = self._train_source_images[start_iter: end_iter]
        #print (batch_bunch)
        batch_names = []
        batch_label = np.array(self._train_source_labels[start_iter: end_iter])
        batch_image = np.zeros(
            [inteval_iter, self._input_h, self._input_w, self._input_c])
        
        for i in range(inteval_iter):
            batch_names.append(batch_bunch[i])
            if self._image_type == 'raw':
                img = cv2.imread(batch_bunch[i], 1)
                #img = cv2.resize(img, (self._input_h, self._input_w))
            elif self._image_type == 'headshot':
                img = load_headshot_image(batch_bunch[i])
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augment = np.random.randint(low = 0, high = 2)
            if augment == 1:
                img = cv2.flip(img,1)
            img = img.astype(np.float32)
            img -= 127.5
            img /= 128.0
            batch_image[i, :, :, :] = img

        return np.array(batch_names), batch_image, batch_label, inteval_iter
        
    def next_batch_target(self):
        if self._is_training and self._batch_train_target_iter % self._num_train_target_batch == 0:
            randnum = random.randint(0, 6666)
            random.seed(randnum)
            random.shuffle(self._train_target_images)
            random.seed(randnum)
            random.shuffle(self._train_target_labels)
            self._batch_train_target_iter = 0
        
        start_iter = self._batch_size * self._batch_train_target_iter
        end_iter = self._batch_size * (self._batch_train_target_iter + 1)
        if end_iter > self._num_train_target_images:
            end_iter = self._num_train_target_images
        inteval_iter = end_iter - start_iter        
        self._batch_train_target_iter += 1
        batch_bunch = self._train_target_images[start_iter: end_iter]
        #print (batch_bunch)
        batch_names = []
        batch_label = np.array(self._train_target_labels[start_iter: end_iter])
        batch_image = np.zeros(
            [inteval_iter, self._input_h, self._input_w, self._input_c])
        for i in range(inteval_iter):
            batch_names.append(batch_bunch[i])
            if self._image_type == 'raw':
                img = cv2.imread(batch_bunch[i], 1)
                #img = cv2.resize(img, (self._input_h, self._input_w))
            elif self._image_type == 'headshot':
                img = load_headshot_image(batch_bunch[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augment = np.random.randint(low = 0, high = 2)
            if augment == 1:
                img = cv2.flip(img,1)
            img = img.astype(np.float32)
            img -= 127.5
            img /= 128.0
            batch_image[i, :, :, :] = img
        return np.array(batch_names), batch_image, batch_label, inteval_iter