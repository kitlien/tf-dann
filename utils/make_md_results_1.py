#coding=utf-8

"""测试结果贴在下面，运行程序即可。注意每一行前面都不要有空格。
MaxAcc: 0.99777 @ Thresh=0.73387
TAR: 0.99884 @ FAR=0.100009751, Thresh=0.58392
TAR: 0.98476 @ FAR=0.010006526, Thresh=0.66814
TAR: 0.90899 @ FAR=0.000999152, Thresh=0.72934
TAR: 0.74889 @ FAR=0.000102016, Thresh=0.77366
TAR: 0.49175 @ FAR=0.000010502, Thresh=0.81457
Area Under Curve (AUC): 0.9988967
TAR: 0.99947 @ FAR=0.509075634, Thresh=0.50000
ACC: 0.49803 @ Thresh=0.50000

MaxAcc: 0.99764 @ Thresh=0.72736
TAR: 0.99884 @ FAR=0.100002250, Thresh=0.58747
TAR: 0.98349 @ FAR=0.010005026, Thresh=0.66978
TAR: 0.90074 @ FAR=0.000997652, Thresh=0.72978
TAR: 0.72296 @ FAR=0.000099015, Thresh=0.77363
TAR: 0.54804 @ FAR=0.000009001, Thresh=0.80407
Area Under Curve (AUC): 0.9988173
TAR: 0.99947 @ FAR=0.518734107, Thresh=0.50000
ACC: 0.48851 @ Thresh=0.50000

MaxAcc: 0.99764 @ Thresh=0.72736
TAR: 0.99884 @ FAR=0.100002250, Thresh=0.58747
TAR: 0.98349 @ FAR=0.010005026, Thresh=0.66978
TAR: 0.90074 @ FAR=0.000997652, Thresh=0.72978
TAR: 0.72296 @ FAR=0.000099015, Thresh=0.77363
TAR: 0.54804 @ FAR=0.000009001, Thresh=0.80407
Area Under Curve (AUC): 0.9988173
TAR: 0.99947 @ FAR=0.518734107, Thresh=0.50000
ACC: 0.48851 @ Thresh=0.50000
"""

"""
@Description: 将上面若干行的测试结果转成测试报告需要的格式
|TAR@FAR=0.1|TAR@FAR=0.01|TAR@FAR=0.001|TAR/FAR/ACC@Thresh=0.5|AUC|。
"""

import os

pwd = os.getcwd()
fpath = os.path.join(pwd, 'make_md_results_1.py')

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
