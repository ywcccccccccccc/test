# coding=utf-8
from matplotlib import pyplot as plt
plt.style.use("ggplot")
import h5py
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
## 环境配置
tf.compat.v1.get_default_graph
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
# 使用GUP跑
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True) #每个gpu占用0.9的显存
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.compat.v1.Session(config=config)#如果电脑有多个GPU，tensorflow默认全部使用。如果想只使用部分GPU，可以设置CUDA_VISIBLE_DEVICES。
os.environ["KERAS_BACKEND"] = "tensorflow"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load model
model = model_from_json(json.load(open("model_struct_1_R.json")))

# load wights
model.load_weights("./model_weights_1_R.h5")
              
# test begin
nClass = 4
seqLen = 128
EsNoLow = 0
EsNoHigh = 18
Gap = 3
pred_mat = np.zeros(((EsNoHigh-EsNoLow)//Gap+1, nClass, nClass))#已修改过代码//
# load test data

for i in range(0, nClass):
    ########打开文件#######
    filename = '../test_data/data'+str(i) + '.h5'#数据集地址
    filename1 = '../test_data/label'+str(i) + '.h5'  # 标签集地址
    filename2 = '../test_data/SNR'+str(i) + '.h5'  # 信噪比集地址

    print(filename)
    print(filename1)
    print(filename2)
    f = h5py.File(filename, 'r')
    f1 = h5py.File(filename1, 'r')
    f2 = h5py.File(filename2, 'r')
    ########读取数据#######
    ####以上步骤通过matlab存成h5文件，存成三维，x为训练数据，y为训练数据标签，z为信噪比。
    X_data = f['X'][:, :]
    Y_data = f1['Y'][:, :]
    Z_data = f2['Z'][:, :]
    f.close()
    f1.close()
    f2.close()
    #########分割训练集和测试集#########
    #每读取到一个数据文件就直接分割为训练集和测试集，防止爆内存
    if i == 0:
        X_train = X_data
        Y_train = Y_data
        Z_train = Z_data
    else:
        X_train = np.vstack((X_train, X_data))
        Y_train = np.vstack((Y_train, Y_data))
        Z_train = np.vstack((Z_train, Z_data))

EsNoArray = Z_train
x_test = X_train
y_test = Y_train
y_predict = model.predict(x_test)

# get predict result
for i in range(y_test.shape[0]):
    axis_0 = (int)((EsNoArray[i] - EsNoLow)/Gap)
    # should be
    axis_1 = (int)(y_test[i])
    # predict to be
    axis_2 = np.argmax(y_predict[i,:])
    pred_mat[axis_0, axis_1, axis_2] = pred_mat[axis_0, axis_1, axis_2] + 1
saveFileName = "pred_confusion_mat_L" + str(seqLen)
np.save(saveFileName, pred_mat)
