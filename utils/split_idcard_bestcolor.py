import os

dir = '/storage-server1/workspace/lianjie/data/new_module_image/224_new_mode_image_0603'
dirs = os.listdir(dir)
new_module_image = open('new_module_image_0603.txt', 'w')
ii = 0
for line in dirs:
    image_dir = os.listdir(os.path.join(dir, line))
    for image in image_dir:
        if image.split('-')[1] == 'idcardimage.jpg':
            ii = ii + 1
        else:
            image_path = os.path.join(dir, line)
            image_path = os.path.join(image_path, image)
            new_module_image.write(image_path)
            new_module_image.write('\n')
new_module_image.close()
print (ii)