import _init_paths
import diluface.utils as utils
import os
import random

save_dir = '/storage-server1/workspace/lianjie/work/source/DiluFaceTrain/diluface_dann/trained_models'
data_dirs = '/data/datagate_bestframe_1200w/'


new_module_pkl = 'new_module_image_0603.pkl'
class_dict_live = utils.load_class_dict(new_module_pkl)
print('--------------------------------------')
utils.print_class_dict_info(class_dict_live, 'Before filter')

train_cache = os.path.join(save_dir, 'trainCache.pkl')
class_dict = utils.load_class_dict_maybe(train_cache, data_dirs, None, None)
print('--------------------------------------')
utils.print_class_dict_info(class_dict, 'Before filter')

old_module_2w_most = '../datagate/old_module_2w_most.pkl'
class_dict_2w_most = utils.load_class_dict_maybe(old_module_2w_most, data_dirs, None, None)
print('--------------------------------------')
utils.print_class_dict_info(class_dict_2w_most, 'Before filter')

#find same id in old_module and new_module data
source_same_list_file = 'file_list/old_new_same_id_in_source.txt'
target_same_list_file = 'file_list/old_new_same_id_in_target.txt'
source_target_mix_file = 'file_list/old_new_same_id_in_source_target_mix.txt'
target_test_list_file = open('file_list/test_target.txt','w')
com_id_list = open('file_list/old_new_same_id.txt', 'w')
source_train_list, target_train_list, source_target_mix = [],[],[]
same_class_dict = {}
k = 0 
for i,key in enumerate(class_dict_live):
    if class_dict.has_key(key):
        for name in class_dict[key]:
            same_class_dict.setdefault(key, []).append(name)
            source_train_list.append('{},{}'.format(name, k))
            source_target_mix.append('{},{}'.format(name, k))
        for name in class_dict_live[key]:
            target_train_list.append('{},{}'.format(name, k))
            source_target_mix.append('{},{}'.format(name, k))
        k = k+1
        com_id_list.write(key)
        com_id_list.write('\n')
    else:
        for name in class_dict_live[key]:
            target_test_list_file.write(name)
            target_test_list_file.write('\n')
print (len(same_class_dict))

#make source train set
for i,key in enumerate(class_dict_2w_most):
    if k<20000 and same_class_dict.has_key(key) == False:
        for name in class_dict_2w_most[key]:
            source_train_list.append('{},{}'.format(name, k))
            source_target_mix.append('{},{}'.format(name, k))
        k = k+1
        
randnum = random.randint(0, 6666)
random.seed(randnum)
random.shuffle(source_train_list)
random.shuffle(target_train_list)
random.shuffle(source_target_mix)
utils.write_list_file(source_same_list_file, source_train_list)
utils.write_list_file(target_same_list_file, target_train_list)
utils.write_list_file(source_target_mix_file, source_target_mix)
target_test_list_file.close()