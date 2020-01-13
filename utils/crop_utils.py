#coding=utf-8
import os
import time
import errno
from multiprocessing import Pool

def write_list_file(filename, written_list, append_break=True):
    with open(filename, 'w') as f:
        if append_break:
            for item in written_list:
                f.write(str(item) + '\n')
        else:
            for item in written_list:
                f.write(str(item))
            
def write_table_file(filename, written_table, delimiter=None):
    delimiter = delimiter or ','
    
    assert isinstance(written_table, (tuple, list))
    number = len(written_table[0])
    for item in written_table:
        assert isinstance(item, (tuple, list))
        assert len(item) == number, '{} != {}'.format(len(item), number)
        
    record_list = zip(*written_table)
    with open(filename, 'w') as f:
        for record in record_list:
            record_str = [str(item) for item in record]
            f.write(delimiter.join(record_str) + '\n')
            
def makedirs_p(path, mode=0o755):
    if not os.path.exists(path):
        try:
            os.makedirs(path, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    elif not os.path.isdir(path):
        raise IOError("%r exists but is not a directory" % path)
        
def list_dirs(paths):
    """Enhancement on `os.listdir`
    """
    if not isinstance(paths, (tuple, list)):
        paths = [paths]
    name_list = []
    for path in paths:
        path_ex = os.path.expanduser(path)
        for item in os.listdir(path_ex):
            name_list.append(os.path.join(path_ex, item))
    return name_list
    
def change_extname(filename, new_extname):
    filename_no_ext = os.path.splitext(filename)[0]
    if new_extname.startswith('.'):
        return ''.join([filename_no_ext, new_extname]) 
    else:
        return '.'.join([filename_no_ext, new_extname])

def process(cmd_line, filename, log_dir, num_workers):
    makedirs_p(log_dir)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_files = len(lines)
    part_size = (num_files + num_workers - 1) // num_workers
    for i in range(num_workers):
        part_size_actual = min(num_files - i * part_size, part_size)
        part_lines = lines[i * part_size: i * part_size + part_size_actual]
        part_filename = os.path.join(log_dir, 'task_{}_file_list.txt'.format(i))
        write_list_file(part_filename, part_lines, False)

    pool = Pool(processes=num_workers)
    for i in range(num_workers):
        list_filename = os.path.join(log_dir, 'task_{}_file_list.txt'.format(i))
        log_filename = os.path.join(log_dir, 'task_{}.log'.format(i))
        pool.apply_async(cmd_line, (list_filename, log_filename))
    pool.close()
    pool.join()

        