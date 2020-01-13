import shutil
import os
from multiprocessing import Pool
import crop_utils

dir = '../datagate/22w_idcardimage.txt'

image_paths = []
filelist = open(dir).readlines()
for line in filelist:
    image_paths.append(line.strip())
    
dst_path = '/storage-server1/workspace/lianjie/data/new_module_image/datagate_22w_idcard'

def process_image_single(list_filename, result_filename, process_i):
    fid = open(list_filename).readlines()
    ii = 0
    for item in fid:
        ori_path = item.strip().split(' ')[0]
        basename = os.path.basename(ori_path)
        shutil.copy(ori_path,os.path.join(dst_path, basename))
        if ii%1000 == 0:
            print ii, process_i, basename
        ii = ii + 1
    fid.close()    


num_workers = 8
num_files = len(image_paths)
log_dir = './log_dir_copy'
if os.path.isdir(log_dir) == False:
    os.makedirs(log_dir)
part_size = (num_files + num_workers - 1) // num_workers
for i in range(num_workers):
    part_size_actual = min(num_files - i * part_size, part_size)
    part_lines = image_paths[i * part_size: i * part_size + part_size_actual]
    part_filename = os.path.join(log_dir, 'task_{}_file_list.txt'.format(i))
    crop_utils.write_list_file(part_filename, part_lines, True)

pool = Pool(processes=num_workers)
for i in range(num_workers):
    list_filename = os.path.join(log_dir, 'task_{}_file_list.txt'.format(i))
    result_filename = os.path.join(log_dir, 'task_{}_file_list_result.txt'.format(i))
    result = pool.apply_async(process_image_single, (list_filename, result_filename, i, ))
pool.close()
pool.join()