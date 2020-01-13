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
    train_class_dict, val_class_dict = utils.split_class_dict_on_value(filtered_class_dict, [100, 0])
    utils.print_class_dict_info(train_class_dict, 'After filter train')
    print('--------------------------------------')
    utils.print_class_dict_info(val_class_dict, 'After filter validation')
    
    train_list_file = 'train_list_2w_w_idcard.txt'
    val_list_file = 'val_list_2w_w_idcard.txt'
    train_list, val_list = [], []
    for k, (train_label, val_label) in enumerate(zip(train_class_dict, val_class_dict)):
        assert train_label == val_label
        for name in train_class_dict[train_label]:
            train_list.append('{} {}'.format(name, k))
        for name in val_class_dict[val_label]:
            val_list.append('{} {}'.format(name, k))
    random.shuffle(train_list)
    random.shuffle(val_list)
    utils.write_list_file(train_list_file, train_list)
    utils.write_list_file(val_list_file, val_list)

def main():

    pkl_filename = '../datagate/old_module_2w_most.pkl'
    if os.path.isfile(pkl_filename) == False:
        path_vector = []

        # result_filename = '../headshot_landmarks/total_list_2w.txt'
        # path_vector.append(result_filename)
    
        # result_filename = '2w_idcardimage.txt'
        # path_vector.append(result_filename)
        
        result_filename = '../datagate/train_list_source_2k.txt'
        path_vector.append(result_filename)
        print len(path_vector)
        train_list = []
        default_dict = defaultdict(lambda:[])
        class_dict = {}
        for ip in range(len(path_vector)):
            path_list = open(path_vector[ip]).readlines()
            i=0
            for line in path_list:
                filename = line.split(',')[0]
                basename = os.path.basename(filename)
                key = basename.split('_')[0]
                #key = line.split("/")[3]
                #print (key)
                default_dict[key].append(line.split(',')[0])
                #class_dict.setdefault(key, []).append(line.strip())
                class_dict.setdefault(key, []).append(filename)
                if i%10000==0:
                    print i,len(path_list)
                    print key,filename
                i+=1
            print(len(default_dict))
        with open(pkl_filename, 'w') as f:
            pickle.dump(class_dict, f)
    else:
        class_dict = utils.load_class_dict(pkl_filename)
        print('Load class_dict from {}'.format(pkl_filename))
    split_classdict_for_train_eval(class_dict)

    
if __name__ == "__main__":
    main()
