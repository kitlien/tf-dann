import os
file = open('test_target.txt').readlines()
class_dict = {}

for line in file:
    name = line.strip()
    key = os.path.basename(name).split('_')[0]
    class_dict.setdefault(key, []).append(name)
    
probe = open('new_module_probe.txt', 'w')
gallery = open('new_module_gallery.txt', 'w')

for k,key in enumerate(class_dict):
    ii = 0
    for name in class_dict[key]:
        if ii == 0:
            gallery.write(name)
            gallery.write('\n')
        else:
            probe.write(name)
            probe.write('\n')
        ii = ii + 1
probe.close()
gallery.close()
            
            