import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

batch_size = 1
feature = tf.convert_to_tensor(np.array([[0.1,0.2],[0.3,0.4]]))
softmax_out = tf.convert_to_tensor(np.array([[0.1,0.2,0.7],[0.5,0.2,0.3]]))

feature_ex = tf.expand_dims(feature, 2)
softmax_out_ex = tf.expand_dims(softmax_out, 1)
outer_product_out = tf.matmul(feature_ex, softmax_out_ex)

weight = tf.slice(softmax_out, [0,2], [-1, 1])
outer_product_out_test = feature * weight

domain_label = tf.convert_to_tensor(np.array([[1]] * batch_size + [[0]] * batch_size))

#outer_product_out.narrow(2, i, 1).squeeze(2)

outer_product_out_1 = outer_product_out[:,:,0]
outer_product_out_2 = outer_product_out[:,:,1]


pred_t = tf.slice(softmax_out, [0, 0], [-1, 2])

class_weight = tf.reduce_mean(softmax_out, axis = 0)
#class_weight = (class_weight / tf.reduce_max(class_weight))

x = tf.constant(value = [[0.01,0.2,0.3,0.4],[0.002,0.3,0.4,0.5]])
y = tf.where(condition = x>0.01)
new_x = tf.gather(x, y)
mx = x * tf.log(x)
entroy = -tf.reduce_sum(x * tf.log(x))
#loss = tf.nn.softmax_cross_entropy_with_logits(logits = outer_product_out_1, labels = domain_label)
with tf.Session() as sess:
    # print sess.run(outer_product_out_1)
    # print sess.run(outer_product_out_2)
    # print sess.run(weight)
    # print sess.run(outer_product_out_test)
    #print sess.run(loss)
    #print sess.run(class_weight)
    print sess.run(pred_t)
    #print sess.run(y)
    #print sess.run(mx)