#coding=utf-8
# Copyright (c) 2017-2018 Dilusense Inc. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import _init_paths
import diluface.utils as utils
from diluface.evaluator import FaceEvaluator
from common import dap_func_wrapper

def get_1000_identity(filename):
    filename = os.path.basename(filename)
    return filename.split('-')[0]
    
def get_newModule_identity(filename):
    filename = os.path.basename(filename)
    return filename.split('_')[0]
    
def get_diluface_identity(filename):
    filename = os.path.basename(filename)
    return filename[:18].upper()
    
def get_gallery_and_probe(test_name='1000'):
    get_label_func = get_diluface_identity
    if test_name == '1000':
        gallery_root = '/storage-server1/workspace/kangkai/data/test_data/1000face/image_a_dealt/rgb_color'
        probe_root = '/storage-server1/workspace/kangkai/data/test_data/1000face/image_b_dealt/rgb_color' 
        get_label_func = get_1000_identity
    elif test_name in ['400k_clean', '400k-clean']:
        gallery_root = '/storage-server1/workspace/kangkai/data/test_data/datagate_test/40w_datagate_test_a_224_clean/'
        probe_root = '/storage-server1/workspace/kangkai/data/test_data/datagate_test/12k_datagate_test_b_224_clean/'
        #gallery_root = '/data/test_data/datagate_1v40w_20161229/gallery224/'
        #probe_root = '/data/test_data/datagate_1v40w_20161229/probe224/'
        get_label_func = get_diluface_identity
    elif test_name == '400k':
        gallery_root = '/storage-server1/workspace/kangkai/data/test_data/datagate_test/40w_datagate_test_a_224/'
        probe_root = '/storage-server1/workspace/kangkai/data/test_data/datagate_test/12k_datagate_test_b_224/'
        get_label_func = get_diluface_identity
    else:
        raise ValueError('Parameter error')
        
    gallery_list = 'datagate/new_module_test_1vsn/new_module_gallery.txt'
    probe_list = 'datagate/new_module_test_1vsn/new_module_probe.txt'
    
    print('gallery: {}:{}'.format(utils.get_host_ip(), gallery_root))
    print('probe: {}:{}'.format(utils.get_host_ip(), probe_root))
    #gallery_paths = utils.list_dirs(gallery_root, sort=True)
    gallery_paths = []
    for line in open(gallery_list).readlines():
        gallery_paths.append(line.strip())
    gallery_labels = [get_newModule_identity(item) for item in gallery_paths]
    #probe_paths = utils.list_dirs(probe_root, sort=True)
    probe_paths = []
    for line in open(probe_list).readlines():
        probe_paths.append(line.strip())
    probe_labels = [get_newModule_identity(item) for item in probe_paths]
    return gallery_paths, gallery_labels, probe_paths, probe_labels
    
def main(args):
    utils.print_arguments(args)
    print('--------------------------------------')

    dap_func = dap_func_wrapper(args.image_height, args.image_width)
    gallery_paths, gallery_labels, probe_paths, probe_labels = get_gallery_and_probe(args.test_name)
    evaluator = FaceEvaluator(
        devices=args.devices,
        # 数据输入有关的参数
        dap_func=dap_func, 
        num_dap_threads=args.num_dap_threads, 
        batch_size=args.batch_size,
        # 网络模型有关的参数
        model_def=args.model_def, 
        embedding_size=args.embedding_size,
        use_batch_norm=args.use_batch_norm,
        use_normalized=args.use_normalized,
        pretrained_model=args.pretrained_model,
        fusion_method=args.fusion_method)
    evaluator.evaluate_1vsn(probe_paths=probe_paths,
        probe_labels=probe_labels,
        gallery_paths=gallery_paths,
        gallery_labels=gallery_labels,
        top_num=args.top_num,
        save_feature=args.save_feature,
        use_detail=args.use_detail)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--test_name', type=str, default='1000')

    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--num_dap_threads', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--model_def', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_normalized', action='store_true')
    parser.add_argument('--fusion_method', type=str, default='single')
    
    parser.add_argument('--top_num', type=int, default=10)
    parser.add_argument('--save_feature', action='store_true')
    parser.add_argument('--use_detail', action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
