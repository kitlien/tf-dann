import pickle
import os
from collections import defaultdict
import _init_paths
import diluface.utils as utils
import numpy as np
import random

log_dir = './log_dir'
class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def split_classdict_for_train_eval(class_dict):
    print (len(class_dict))
    num_classes = 20000
    num_classes = min(num_classes, len(class_dict))
    filtered_class_dict = utils.sort_class_dict_by_number(class_dict, num_classes)
    np.random.seed(666)
    train_class_dict, val_class_dict = utils.split_class_dict_on_value(filtered_class_dict, [100, 1])
    utils.print_class_dict_info(train_class_dict, 'After filter train')
    print('--------------------------------------')
    utils.print_class_dict_info(val_class_dict, 'After filter validation')
    
    train_list_file = 'train_list_new_moudle.txt'
    val_list_file = 'val_list_new_moudle.txt'
    train_list, val_list = [], []
    for k, (train_label, val_label) in enumerate(zip(train_class_dict, val_class_dict)):
        assert train_label == val_label
        for name in train_class_dict[train_label]:
            train_list.append('{},{}'.format(name, k))
        for name in val_class_dict[val_label]:
            val_list.append('{},{}'.format(name, k))
    random.shuffle(train_list)
    random.shuffle(val_list)
    utils.write_list_file(train_list_file, train_list)
    utils.write_list_file(val_list_file, val_list)

def main():
    pkl_filename = 'new_module_image_test.pkl'
    image_path = '/storage-server1/workspace/lianjie/work/FaceGeneralize/image_undistort/new_image_distort/original'
    image_path_0603 = 'new_module_image_0603.txt'
    class_dict = {}
    if os.path.isfile(pkl_filename) == False:
        path_list = os.listdir(image_path)
        i=0
        for line in path_list:
            basename = os.path.basename(line)
            key = basename.split('_')[0]
            class_dict.setdefault(key, []).append(os.path.join(image_path, line))
            if i%10000 == 0:
                print i,len(path_list)
                print key,line.strip()
            i+=1
        path_list = open(image_path_0603).readlines()
        for line in path_list:
            basename = os.path.basename(line.strip())
            key = basename.split('_')[0]
            class_dict.setdefault(key, []).append(os.path.join(image_path, line.strip()))
            if i%10000 == 0:
                print i,len(path_list)
                print key,line.strip()
            i+=1
        with open(pkl_filename, 'w') as f:
            pickle.dump(class_dict, f)
    else:
        class_dict = utils.load_class_dict(filename)
        print('Load class_dict from {}'.format(filename))
    split_classdict_for_train_eval(class_dict)
    
    # com_id_list = open('new_same_id_12.txt', 'w')
    # for i,key in enumerate(class_dict_1):
        # if class_dict_2.has_key(key):
            # com_id_list.write(key)
            # com_id_list.write('\n')
    # com_id_list.close()
if __name__ == "__main__":
    main()
