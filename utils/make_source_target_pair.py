import os

source_file = open('../primesense/old_new_same_id_in_source.txt').readlines()
target_file = open('../primesense/old_new_same_id_in_target_2000.txt').readlines()

source_target_pair = open('../primesense/source_target_pair_2000.txt', 'w')
ii = 0
for line in source_file:
    source_target_pair.write(line.strip())
    source_target_pair.write(',')
    domain_label = 0
    source_target_pair.write(str(domain_label))
    source_target_pair.write(',')
    source_target_pair.write(target_file[ii].strip())
    source_target_pair.write(',')
    domain_label = 1
    source_target_pair.write(str(domain_label))
    source_target_pair.write('\n')
    ii = ii + 1
    if ii == len(target_file):
        ii = 0
source_target_pair.close()
    

