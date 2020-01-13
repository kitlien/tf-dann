import os

filelist = open('old_new_same_id.txt').readlines()

live_dir = '/storage-server1/workspace/lianjie/work/FaceGeneralize/image_undistort/new_image_distort/original'
idcard_dir = '/storage-server1/workspace/lianjie/work/FaceGeneralize/image_undistort/new_image_distort/idcardimage_crop'

livelist = os.listdir(live_dir)
icardlist = os.listdir(idcard_dir)
file_live = open('new_live_photo_clean.txt','w')
file_idcard = open('new_idcard_photo_clean.txt','w')

for line in livelist:
    liveid = os.path.basename(line).split('_')[0]
    not_in_com = True
    for file in filelist:
        comid = os.path.basename(file).strip().split('_')[0]
        if liveid == comid:
            not_in_com = False
    if not_in_com == True:
        file_live.write(os.path.join(live_dir, line))
        file_live.write('\n')
        
for line in icardlist:
    liveid = os.path.basename(line).split('_')[0]
    not_in_com = True
    for file in filelist:
        comid = os.path.basename(file).strip().split('_')[0]
        if liveid == comid:
            not_in_com = False
    if not_in_com == True:
        file_idcard.write(os.path.join(idcard_dir, line))
        file_idcard.write('\n')
        
file_live.close()
file_idcard.close()