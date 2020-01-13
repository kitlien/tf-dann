import _init_paths
import diluface.utils as utils
import os

save_dir = '/storage-server1/workspace/lianjie/work/source/DiluFaceTrain/diluface_dann/trained_models'
data_dirs = '/data/datagate_bestframe_1200w/'


old_module_2w_pkl = '/storage-server1/workspace/lianjie/work/FaceGeneralize/tf-dann/utils/old_module_400W.pkl'
class_dict_live = utils.load_class_dict(old_module_2w_pkl)
print('--------------------------------------')
utils.print_class_dict_info(class_dict_live, 'Before filter')


train_cache = os.path.join(save_dir, 'trainCache.pkl')
class_dict = utils.load_class_dict_maybe(train_cache, data_dirs, None, None)
print('--------------------------------------')
utils.print_class_dict_info(class_dict, 'Before filter')

train_list_file = open('2w_idcardimage.txt','w')
for k,key in enumerate(class_dict_live):
    if class_dict.has_key(key):
        for name in class_dict[key]:
            basename = os.path.basename(name)
            if name.split('-')[1] == 'idcardimage.jpg':
                train_list_file.write(name)
                train_list_file.write(' ')
                for i in range(12):
                    index = 0
                    train_list_file.write(str(index))
                    if i!=11:
                        train_list_file.write(' ')
                train_list_file.write('\n')
            
