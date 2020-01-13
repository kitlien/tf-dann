import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
model_reader = pywrap_tensorflow.NewCheckpointReader('./pretrained_models/20190416_021402_l2_softmax.ckpt-43000')
var_dict = model_reader.get_variable_to_shape_map()
for key in var_dict:
    print("variable name: ", key)
    #print(model_reader.get_tensor(key))