#coding=utf-8
# Copyright (c) 2017-2019 Dilusense Inc. All Rights Reserved.
import numpy as np
import tensorflow as tf
import _init_paths
import diluface.utils as utils
import os
import cv2

class cameraParam():
    def __init__(self):
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0
        self.k3 = 0.0
        
def align_and_crop_112x96(image, landmarks):
    align_size = (96, 112)
    std_landmarks = [[30.2946, 51.6963], # left eye
                    [65.5318, 51.5014],  # right eye
                    [48.0252, 71.7366],  # nose tip
                    [33.5493, 92.3655],  # left mouth corner
                    [62.7299, 92.2041]]  # right mouth corner
    landmarks = np.asarray(landmarks).reshape(5, 2)
    image_cropped = utils.align_and_crop(image, landmarks, std_landmarks, align_size)
    return image_cropped
    
def align_and_crop_130x110(image, landmarks):
    align_size = (110, 130)
    std_landmarks = [[37.2946, 60.6963], # left eye
                    [72.5318, 60.5014],  # right eye
                    [55.0252, 80.7366],  # nose tip
                    [40.5493, 101.3655], # left mouth corner
                    [69.7299, 101.2041]] # right mouth corner
    landmarks = np.asarray(landmarks).reshape(5, 2)
    image_cropped = utils.align_and_crop(image, landmarks, std_landmarks, align_size)
    return image_cropped
    
def align_and_crop_224x224(image, landmarks):
    align_size = (256, 256)
    crop_size = (224, 224)
    std_landmarks = [[85.49595008, 85.73911040000002 + 30],  # left eye
                    [168.39541376, 84.30362432000001 + 30],  # right eye
                    [127.07756863999998, 136.76120383999998 + 30],  # nose tip
                    [90.34349120000002, 174.31525376000002 + 30],  # left mouth corner
                    [166.65585920000004, 172.99233343999998 + 30]]  # right mouth corner
    landmarks = np.asarray(landmarks).reshape(5, 2)
    crop_center = ((landmarks[0, 0] + landmarks[1, 0]) / 2, (landmarks[0, 1] + landmarks[1, 1]) / 2)
    image_cropped = utils.align_and_crop(image, landmarks, std_landmarks, align_size, crop_size, crop_center)
    return image_cropped

def add_old_distort_headshot(src, param, tlx, tly):
    
    h_src, w_src, _ = src.shape
    copy_top = tly
    copy_bottom = 1080 - h_src - tly
    copy_left = tlx
    copy_right = 1920 - w_src - tlx
    #print (copy_top, copy_bottom, copy_left, copy_right)
    pre_crop_image = cv2.copyMakeBorder(src,copy_top,copy_bottom,copy_left,copy_right,cv2.BORDER_CONSTANT,value=[0,0,0])
    h, w, c = pre_crop_image.shape
    dst = np.zeros(shape=(h, w , c), dtype=np.uint8)
    I = np.arange(tlx, tlx + w_src)
    J = np.arange(h)
    
    X = (I-param.cx)/param.fx
    Y = (J-param.cy)/param.fy
    
    for j in range(tly, tly + h_src, 1):
        
        r2 = X*X + Y[j]*Y[j]
        
        #add distort with all distortion params
        newX = X*(1 + param.k1 * r2 + param.k2*r2*r2) + 2*param.p1*X*Y[j] + param.p2 * (r2 + 2*X*X)
        newY = Y[j]*(1 + param.k1 * r2 + param.k2*r2*r2) + 2*param.p2*X*Y[j] + param.p1 * (r2 + 2*Y[j]*Y[j])
        
        #convert to image pixel coordinate
        u = newX*param.fx + param.cx
        v = newY*param.fy + param.cy
        
        #bi-linear interpolate
        u0 = np.floor(u)
        v0 = np.floor(v)
        
        u0 = u0.astype(np.int32)
        v0 = v0.astype(np.int32)
        u1 = u0+1
        v1 = v0+1
        
        #fill Null value with zero
        idx1 = (u0<0) + (u0>=w-2)
        idx2 = (v0<0) + (v0>=h-2)
        idx3 = (u1<0) + (u1>=w-1)
        idx4 = (v1<0) + (v1>=h-1)
        u0[idx1 + idx2] = 0
        v0[idx1 + idx2] = 0
        u1[idx3 + idx4] = 0
        v1[idx3 + idx4] = 0
        src[0,0,:] = 0
        
        dx = u-u0
        dy = v-v0
        w1 = (1-dx)*(1-dy)
        w2 = dx*(1-dy)
        w3 = (1-dx)*dy
        w4 = dx*dy
        
        for x in range(c):
            dst[j,tlx:tlx + w_src,x] = w1 * pre_crop_image[v0,u0,x] + \
                        w2 * pre_crop_image[v1,u0,x] + \
                        w3 * pre_crop_image[v0,u1,x] + \
                        w4 * pre_crop_image[v1,u1,x]
    return dst 

def add_old_distort_headshot_with_map(src, param, h_src, w_src, tlx, tly):
    
    h, w, c = src.shape
    dst = np.zeros(shape=(h, w, c), dtype=np.uint8)
    map1 = np.zeros(shape=(h, w), dtype=np.float32)
    map2 = np.zeros(shape=(h, w), dtype=np.float32)
    I = np.arange(tlx, tlx + w_src)
    J = np.arange(h)
    
    X = (I-param.cx)/param.fx
    Y = (J-param.cy)/param.fy
    
    for j in range(tly, tly + h_src, 1):
        
        r2 = X*X + Y[j]*Y[j]
        
        #add distort with all distortion params
        kr = 1 + param.k1 * r2 + param.k2*r2*r2
        newX = X*kr + 2*param.p1*X*Y[j] + param.p2 * (r2 + 2*X*X)
        newY = Y[j]*kr + 2*param.p2*X*Y[j] + param.p1 * (r2 + 2*Y[j]*Y[j])
        
        #convert to image pixel coordinate
        u = newX*param.fx + param.cx
        v = newY*param.fy + param.cy
        
        #bi-linear interpolate
        u0 = np.floor(u)
        v0 = np.floor(v)
        
        map1[j,tlx:tlx + w_src] = u
        map2[j,tlx:tlx + w_src] = v
    return map1,map2
    
def point_undistort_project(ptx, pty, param):
    i = ptx
    j = pty
    X = (i-param.cx)/param.fx
    Y = (j-param.cy)/param.fy
    r2 = X*X + Y*Y
    #add distort
    newX = X*(1 + param.k1 * r2 + param.k2*r2*r2 + param.k3*r2*r2*r2) + 2*param.p1*X*Y + param.p2 * (r2 + 2*X*X)
    newY = Y*(1 + param.k1 * r2 + param.k2*r2*r2 + param.k3*r2*r2*r2) + 2*param.p2*X*Y + param.p1 * (r2 + 2*Y*Y)
    #convert to image pixel coordinate
    u = newX*param.fx + param.cx
    v = newY*param.fy + param.cy
    # get new coord
    new_ptx = u
    new_pty = v
    #print (i,j,u,v)
    return new_ptx, new_pty
    
def align_and_crop_224x224_with_undistort(image, landmarks, tlx, tly):
    param_old = cameraParam()
    param_old.fx = 1033.72
    param_old.fy = 1034.34
    param_old.cx = 914.985
    param_old.cy = 522.985
    param_old.k1 = -0.251142
    param_old.k2 = 0.0742488
    param_old.p1 = -0.000100508
    param_old.p2 = 0.000274961
    param_old.k3 = 0
    
    landmarks = np.asarray(landmarks).reshape(5, 2)
    #undistort_src = add_old_distort_headshot(image, param_old, tlx, tly)
    
    h_src, w_src, _ = image.shape
    copy_top = tly
    copy_bottom = 1080 - h_src - tly
    copy_left = tlx
    copy_right = 1920 - w_src - tlx
    pre_crop_image = cv2.copyMakeBorder(image,copy_top,copy_bottom,copy_left,copy_right,cv2.BORDER_CONSTANT,value=[0,0,0])

    map1, map2 = add_old_distort_headshot_with_map(pre_crop_image, param_old, h_src, w_src, tlx, tly)
    undistort_src = cv2.remap(pre_crop_image, map1, map2, cv2.INTER_CUBIC)
    
    param_old.k1 = 0.251142
    param_old.k2 = -0.0742488
    param_old.p1 = 0.000100508
    param_old.p2 = -0.000274961
    undistort_landmarks = np.zeros((5,2), dtype = np.float32)
    for i in range(5):
       ptx, pty = point_undistort_project(landmarks[i][0], landmarks[i][1], param_old)
       #cv2.circle(undistort_src,(int(ptx),int(pty)),1,(0,0,255),4)
       undistort_landmarks[i][0] = ptx
       undistort_landmarks[i][1] = pty
    image_cropped = align_and_crop_224x224(undistort_src, undistort_landmarks)
    return image_cropped

def align_and_crop_224x224_with_coord(image, landmarks, tlx, tly):
    align_size = (256, 256)
    crop_size = (224, 224)
    std_landmarks = [[85.49595008, 85.73911040000002 + 30],  # left eye
                    [168.39541376, 84.30362432000001 + 30],  # right eye
                    [127.07756863999998, 136.76120383999998 + 30],  # nose tip
                    [90.34349120000002, 174.31525376000002 + 30],  # left mouth corner
                    [166.65585920000004, 172.99233343999998 + 30]]  # right mouth corner
    landmarks = np.asarray(landmarks).reshape(5, 2)
    for i in range(5):
        landmarks[i][0] = landmarks[i][0] - tlx
        landmarks[i][1] = landmarks[i][1] - tly
    crop_center = ((landmarks[0, 0] + landmarks[1, 0]) / 2, (landmarks[0, 1] + landmarks[1, 1]) / 2)
    image_cropped = utils.align_and_crop(image, landmarks, std_landmarks, align_size, crop_size, crop_center)
    return image_cropped

def align_and_crop_224x224_with_rand_undistort(image, landmarks, tlx, tly):
    augment = np.random.randint(low = 0, high = 2)
    if augment == 1:
        image_cropped = align_and_crop_224x224_with_undistort(image, landmarks, tlx, tly)
    else:
        image_cropped = align_and_crop_224x224_with_coord(image, landmarks, tlx, tly)
    return image_cropped
    
    
def align_and_crop_320x250(image, landmarks):
    align_size = (250, 320)
    std_landmarks = [[85.49595008*0.9802594866071432, 85.73911040000002*1.247223772321429 + 20 * 1.247223772321429],  # left eye
                    [168.39541376*0.9802594866071432, 84.30362432000001*1.247223772321429 + 20 * 1.247223772321429],  # right eye
                    [127.07756863999998*0.9802594866071432, 136.76120383999998*1.247223772321429 + 10 * 1.247223772321429],  # nose tip
                    [90.34349120000002*0.9802594866071432, 174.31525376000002*1.247223772321429 + 10 * 1.247223772321429],  # left mouth corner
                    [166.65585920000004*0.9802594866071432, 172.99233343999998*1.247223772321429 + 10 * 1.247223772321429]]  # right mouth corner
    landmarks = np.asarray(landmarks).reshape(5, 2)
    image_cropped = utils.align_and_crop(image, landmarks, std_landmarks, align_size)
    return image_cropped
    
def align_and_crop_224x224_with_chin(image, landmarks):
    align_size = (256, 256)
    crop_size = (224, 224)
    std_landmarks = [[85.49595008, 85.73911040000002],  # left eye
                    [168.39541376, 84.30362432000001],  # right eye
                    [127.07756863999998, 136.76120383999998],  # nose tip
                    [90.34349120000002, 174.31525376000002],  # left mouth corner
                    [166.65585920000004, 172.99233343999998]]  # right mouth corner
    landmarks = np.asarray(landmarks).reshape(5, 2)
    crop_center = (landmarks[2, 0], landmarks[2, 1] * 0.95)
    image_cropped = utils.align_and_crop(image, landmarks, std_landmarks, align_size, crop_size, crop_center)
    return image_cropped
    
def is_idcardimage(filename):
    basename = os.path.basename(filename)
    if (basename.split('-')[1] == 'idcardimage.jpg'):
        return True
    else:
        return False
        
def dap_func_wrapper(crop_type):
    """ DAP, decode, augment and preprocess
        先进行数据增强, 再进行预处理
    """
    def dap_func(decoded_record, augment=False):
        num_channels = 3
        filename = decoded_record[0]
        idcard = tf.py_func(is_idcardimage, [filename], tf.bool)
        
        content = tf.read_file(filename)
        image = tf.image.decode_image(content, channels=num_channels)
        
        image_src = image
        
        tlx = decoded_record[1]
        tly = decoded_record[2]
        landmarks = decoded_record[3:13]
        #image = tf.image.decode_jpeg(content, channels=num_channels)
        landmarks = tf.convert_to_tensor(landmarks)
        tlx = tf.convert_to_tensor(tlx)
        tly = tf.convert_to_tensor(tly)
        print(landmarks)
        if crop_type == '224x224':
            image_height, image_width = 224, 224
            image = tf.py_func(align_and_crop_224x224, [image, landmarks], tf.uint8)
        elif crop_type == '130x110':
            image_height, image_width = 130, 110
            image = tf.py_func(align_and_crop_130x110, [image, landmarks], tf.uint8)
        elif crop_type == '112x96':
            image_height, image_width = 112, 96
            image = tf.py_func(align_and_crop_112x96, [image, landmarks], tf.uint8)
        elif crop_type == '112x96_random_crop':
            image_height, image_width = 112, 96
            image = tf.py_func(align_and_crop_130x110, [image, landmarks], tf.uint8)
        elif crop_type == '320x250':
            image_height, image_width = 320, 250
            image = tf.py_func(align_and_crop_320x250, [image, landmarks], tf.uint8)
        elif crop_type == '299x235_random_crop':
            image_height, image_width = 299, 235
            image = tf.py_func(align_and_crop_320x250, [image, landmarks], tf.uint8)
        elif crop_type == '224x224_with_chin':
            image_height, image_width = 224, 224
            image = tf.py_func(align_and_crop_224x224_with_chin, [image, landmarks], tf.uint8)
        elif crop_type == '224x224_with_coord':
            image_height, image_width = 224, 224
            image = tf.py_func(align_and_crop_224x224_with_coord, [image, landmarks, tlx, tly], tf.uint8)
        elif crop_type == '224x224_with_undistort':
            image_height, image_width = 224, 224
            image = tf.py_func(align_and_crop_224x224_with_undistort, [image, landmarks, tlx, tly], tf.uint8)
        elif crop_type == '224x224_with_rand_undistort':
            image_height, image_width = 224, 224
            image = tf.py_func(align_and_crop_224x224_with_rand_undistort, [image, landmarks, tlx, tly], tf.uint8)

        else:
            raise ValueError('Unsupported crop_type')
            
        image = tf.cond(idcard, lambda: image_src, lambda: image)
        image = tf.cast(image, tf.float32)
        # augment
        if augment:
            image = tf.random_crop(image, [image_height, image_width, num_channels])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)
            
        if augment:
            image = tf.image.random_flip_left_right(image)
        if augment:
            stddev = tf.random_uniform((1,), minval=1.0, maxval=4.0)[0]
            noise = tf.random_normal([image_height, image_width, num_channels], mean=0.0, stddev=stddev, dtype=tf.float32)
            image = tf.minimum(tf.maximum(image + noise, 0), 255)
            # bug fix
            image = tf.minimum(tf.maximum(tf.image.random_brightness(image, max_delta=20), 0), 255)
            image = tf.minimum(tf.maximum(tf.image.random_contrast(image, lower=0.6, upper=1.4), 0), 255)
        # preprocess
        image -= 127.5
        image /= 128.0
        
        # # set shape, required
        image.set_shape([image_height, image_width, num_channels])
        
        return image
        
    return dap_func
    
    
def dap_func_wrapper_for_test(image_height, image_width):
    """ DAP, decode, augment and preprocess
    """
    def dap_func(content, augment=False):
        # decode
        image = tf.image.decode_image(content, channels=3)
        image = tf.cast(image, tf.float32)
        
        # augment
        if augment:
            image = tf.random_crop(image, [image_height, image_width, 3])
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)
        # preprocess
        image -= 127.5
        image /= 128.0
        
        # set shape, required
        image.set_shape([image_height, image_width, 3])
        
        return image
        
    return dap_func