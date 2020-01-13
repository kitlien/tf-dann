#coding=utf-8

"""测试结果贴在下面，运行程序即可。注意每一行前面都不要有空格。
MaxAcc: 0.99780 @ Thresh=0.74037
TAR: 0.99894 @ FAR=0.099982747, Thresh=0.59284
TAR: 0.98794 @ FAR=0.010003526, Thresh=0.67922
TAR: 0.91206 @ FAR=0.000996152, Thresh=0.73803
TAR: 0.72466 @ FAR=0.000100515, Thresh=0.78231
TAR: 0.56138 @ FAR=0.000010502, Thresh=0.80882
Area Under Curve (AUC): 0.9989347
TAR: 0.99947 @ FAR=0.530063835, Thresh=0.50000
ACC: 0.47734 @ Thresh=0.50000

MaxAcc: 0.99804 @ Thresh=0.74337
TAR: 0.99894 @ FAR=0.100026254, Thresh=0.59119
TAR: 0.99048 @ FAR=0.009994524, Thresh=0.67808
TAR: 0.92783 @ FAR=0.000999152, Thresh=0.73895
TAR: 0.76222 @ FAR=0.000100515, Thresh=0.78298
TAR: 0.57111 @ FAR=0.000010502, Thresh=0.81457
Area Under Curve (AUC): 0.9990987
TAR: 0.99947 @ FAR=0.526877349, Thresh=0.50000
ACC: 0.48048 @ Thresh=0.50000

MaxAcc: 0.99817 @ Thresh=0.73087
TAR: 0.99894 @ FAR=0.099984248, Thresh=0.58807
TAR: 0.98730 @ FAR=0.010006526, Thresh=0.67269
TAR: 0.91439 @ FAR=0.001000653, Thresh=0.73271
TAR: 0.73333 @ FAR=0.000100515, Thresh=0.77872
TAR: 0.56772 @ FAR=0.000009001, Thresh=0.80707
Area Under Curve (AUC): 0.9989291
TAR: 0.99958 @ FAR=0.514345938, Thresh=0.50000
ACC: 0.49284 @ Thresh=0.50000

MaxAcc: 0.99817 @ Thresh=0.73087
TAR: 0.99894 @ FAR=0.099976746, Thresh=0.58563
TAR: 0.99238 @ FAR=0.009997525, Thresh=0.66949
TAR: 0.93661 @ FAR=0.000996152, Thresh=0.72825
TAR: 0.78571 @ FAR=0.000100515, Thresh=0.77397
TAR: 0.59460 @ FAR=0.000010502, Thresh=0.80557
Area Under Curve (AUC): 0.9991335
TAR: 0.99958 @ FAR=0.514345938, Thresh=0.50000
ACC: 0.49284 @ Thresh=0.50000

MaxAcc: 0.99783 @ Thresh=0.75738
TAR: 0.99894 @ FAR=0.099964745, Thresh=0.59897
TAR: 0.98730 @ FAR=0.009999025, Thresh=0.69200
TAR: 0.91196 @ FAR=0.001002153, Thresh=0.75301
TAR: 0.73196 @ FAR=0.000100515, Thresh=0.79694
TAR: 0.56889 @ FAR=0.000009001, Thresh=0.82208
Area Under Curve (AUC): 0.9989367
TAR: 0.99947 @ FAR=0.534355989, Thresh=0.50000
ACC: 0.47311 @ Thresh=0.50000
"""

"""
@Description: 将上面若干行的测试结果转成测试报告需要的格式
|TAR@FAR=0.1|TAR@FAR=0.01|TAR@FAR=0.001|TAR/FAR/ACC@Thresh=0.5|AUC|。
"""

import os

pwd = os.getcwd()
fpath = os.path.join(pwd, 'make_md_results.py')

print('|模型\指标|TAR@FAR=0.1|TAR@FAR=0.01|TAR@FAR=0.001|TAR@FAR=0.0001|TAR@FAR=0.00001|')
print('|:--:|--|--|--|--|--|')

with open(fpath) as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if 'MaxAcc' in line and 34 == len(line):
            tar1 = lines[i + 1][5:12]
            t1 = lines[i + 1][39:46]
            tar2 = lines[i + 2][5:12]
            t2 = lines[i + 2][39:46]
            tar3 = lines[i + 3][5:12]
            t3 = lines[i + 3][39:46]
            tar4 = lines[i + 4][5:12]
            t4 = lines[i + 4][39:46]
            tar5 = lines[i + 5][5:12]
            t5 = lines[i + 5][39:46]
            i += 9

            print('|Model|%s(%s)|%s(%s)|%s(%s)|%s(%s)|%s(%s)|' % (tar1, t1, tar2, t2, tar3, t3, tar4, t4, tar5, t5))
