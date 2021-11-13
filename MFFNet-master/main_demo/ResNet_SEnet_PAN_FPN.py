import h5py
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.models as Model
from tensorflow.keras.layers import Input, Reshape, Conv2D, Flatten, Dense, Activation, MaxPooling2D, AlphaDropout, AveragePooling2D, BatchNormalization, UpSampling2D
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
## 环境配置
tf.compat.v1.get_default_graph
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
# 使用GUP运行
gpu_options = tf.GPUOptions(allow_growth=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True) #每个gpu占用0.9的显存
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)#如果电脑有多个GPU，tensorflow默认全部使用。如果想只使用部分GPU，可以设置CUDA_VISIBLE_DEVICES。
os.environ["KERAS_BACKEND"] = "tensorflow"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
############################################
epochs_num = 50
Floor = 3
L = 128
for i in range(0, 4):
    ########打开文件#######
    filename = 'C:/Users/Administrator/Desktop/CL_project/CL_project/test_data' + str(i) + '.h5'  # 数据集地址
    filename1 = 'C:/Users/Administrator/Desktop/CL_project/CL_project/test_data' + str(i) + '.h5'  # 标签集地址
    filename2 = 'C:/Users/Administrator/Desktop/CL_project/CL_project/test_data' + str(i) + '.h5'  # 信噪比集地址
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
    n_examples = X_data.shape[0]
    n_train = int(n_examples * 0.85)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)#随机选取训练样本下标
    # train_idx = range(0, n_examples)# 随机选取训练样本下标
    test_idx = list(set(range(0, n_examples))-set(train_idx))        #测试样本下标
    if i == 0:
        X_train = X_data[train_idx]
        Y_train = Y_data[train_idx]
        Z_train = Z_data[train_idx]
        X_test = X_data[test_idx]
        Y_test = Y_data[test_idx]
        Z_test = Z_data[test_idx]
    else:
        X_train = np.vstack((X_train, X_data[train_idx]))
        Y_train = np.vstack((Y_train, Y_data[train_idx]))
        Z_train = np.vstack((Z_train, Z_data[train_idx]))
        X_test = np.vstack((X_test, X_data[test_idx]))
        Y_test = np.vstack((Y_test, Y_data[test_idx]))
        Z_test = np.vstack((Z_test, Z_data[test_idx]))
print('Dimension of training set X：', X_train.shape)
print('Dimension of training set Y：', Y_train.shape)
print('Dimension of training set Z：', Z_train.shape)
print('Dimension of test set X：', X_test.shape)
print('Dimension of test set Y：', Y_test.shape)
print('Dimension of test set Z：', Z_test.shape)

"""建立模型"""
classes = ['Num_Sig_1', 'Num_Sig_2', 'Num_Sig_3', 'Num_Sig_4']
data_format = 'channels_first'
# 以上的标签需要与MATLAB所输出的标签打成一致

# 定义每个epoch的训练时间
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def residual_stack(Xm, filter_num, kennel_size,Seq,pool_size):
    #1*1 Conv Linear
    Xm = tf.keras.layers.Conv2D(filter_num, (1, 1), padding='same', name=Seq+"_conv1", kernel_initializer='glorot_normal',data_format=data_format, kernel_regularizer=tf.keras.regularizers.l2(0.03))(Xm)
    #Residual Unit 1
    Xm_shortcut = Xm
    Xm = tf.keras.layers.Conv2D(filter_num, kennel_size, padding='same', activation="relu",name=Seq+"_conv2", kernel_initializer='glorot_normal', data_format=data_format, kernel_regularizer=tf.keras.regularizers.l2(0.03))(Xm)
    Xm = tf.keras.layers.Conv2D(filter_num, kennel_size, padding='same', name=Seq+"_conv3", kernel_initializer='glorot_normal', data_format=data_format, kernel_regularizer=tf.keras.regularizers.l2(0.03))(Xm)

    Xm = tf.keras.layers.add([Xm, Xm_shortcut])
    Xm = Activation("relu")(Xm)
    Xm = tf.keras.layers.BatchNormalization(axis=1, scale=True)(Xm)
    #Residual Unit 2
    Xm_shortcut = Xm
    Xm = tf.keras.layers.Conv2D(filter_num, kennel_size, padding='same', activation="relu", name=Seq+"_conv4", kernel_initializer='glorot_normal', data_format=data_format, kernel_regularizer=tf.keras.regularizers.l2(0.03))(Xm)
    Xm = tf.keras.layers.Conv2D(filter_num, kennel_size, padding='same', name=Seq + "_conv5", kernel_initializer='glorot_normal', data_format=data_format, kernel_regularizer=tf.keras.regularizers.l2(0.03))(Xm)

    Xm = tf.keras.layers.add([Xm, Xm_shortcut]) # 网络分别处理后的融合
    Xm = Activation("relu")(Xm)
    Xm = tf.keras.layers.BatchNormalization(axis=1, scale=True)(Xm)
    #MaxPooling
    Xm = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format=data_format)(Xm)
    return Xm

## FPN类型
def FPN(Xm, filter_num, UP_size, UP_name):#定义残差块
    Xm = tf.keras.layers.Conv2D(filter_num, (1, 1), name='line'+UP_name)(Xm)
    Xm = tf.keras.layers.UpSampling2D(size=UP_size, name=UP_name)(Xm)
    return Xm
## PAN类型
def PAN(Xm, filter_num, kennel_size, Conv_name, pool_size, strides, upsample=True):# 定义PAN中的GAU，每次调用时需要对卷积核大小进行定义，以及条件语句的限定
    if upsample:
        Xm = tf.keras.layers.Conv2D(filter_num, kennel_size, padding='same', activation="relu", name=Conv_name)(Xm)
        Xm = tf.keras.layers.BatchNormalization(axis=1, scale=True)(Xm)
        Xm = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)(Xm)
    else:
        Xm = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=1)(Xm)
        Xm = tf.keras.layers.Conv2D(filter_num, kennel_size, padding='same', activation="relu", name=Conv_name)(Xm)
        Xm = tf.keras.layers.Conv2D(filter_num, (1, 1), padding='same', activation="relu")(Xm)
        Xm = tf.keras.layers.BatchNormalization(axis=1, scale=True)(Xm)
        Xm = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)(Xm)
    return Xm
in_shp = X_train.shape[1:]   #每个样本的维度[1024,2]
#input layer
Xm_input = Input(in_shp)
# Xm = Xm_input
Xm = Reshape([1, L, 10], input_shape=in_shp)(Xm_input)
#Residual Stack
Xm = residual_stack(Xm, 32, kennel_size=(2, 2), Seq="ReStk0", pool_size=(2, 1))#shape:(512,1,32)，卷积核为3×2，代表每次维度递减一半，列数减少为1
squeeze = tf.keras.layers.GlobalAveragePooling2D()(Xm)
excitation = Dense(16, activation='relu', kernel_initializer='glorot_normal')(squeeze)
excitation = Dense(32, kernel_initializer='glorot_normal')(excitation)
excitation = Activation('sigmoid')(excitation)
excitation = Reshape([32, 1, 1])(excitation)
Xm = Xm*excitation
Xm_R1 = Xm
Xm = residual_stack(Xm, 32, kennel_size=(2, 2), Seq="ReStk1", pool_size=(2, 1))#shape:(256,1,32)，
squeeze = tf.keras.layers.GlobalAveragePooling2D()(Xm)
excitation = Dense(16, activation='relu', kernel_initializer='glorot_normal')(squeeze)
excitation = Dense(32, kernel_initializer='glorot_normal')(excitation)
excitation = Activation('sigmoid')(excitation)
excitation = Reshape([32, 1, 1])(excitation)
Xm = Xm*excitation
Xm_R2 = Xm
Xm = residual_stack(Xm, 32, kennel_size=(2, 2), Seq="ReStk2", pool_size=(2, 1))#shape:(128,1,32)
squeeze = tf.keras.layers.GlobalAveragePooling2D()(Xm)
excitation = Dense(16, activation='relu', kernel_initializer='glorot_normal')(squeeze)
excitation = Dense(32, kernel_initializer='glorot_normal')(excitation)
excitation = Activation('sigmoid')(excitation)
excitation = Reshape([32, 1, 1])(excitation)
Xm = Xm*excitation
Xm_R3 = Xm
Xm = residual_stack(Xm, 32, kennel_size=(2, 2), Seq="ReStk3", pool_size=(2, 1))#shape:(64,1,32)
squeeze = tf.keras.layers.GlobalAveragePooling2D()(Xm)
excitation = Dense(16, activation='relu', kernel_initializer='glorot_normal')(squeeze)
excitation = Dense(32, kernel_initializer='glorot_normal')(excitation)
excitation = Activation('sigmoid')(excitation)
excitation = Reshape([32, 1, 1])(excitation)
Xm = Xm*excitation
Xm_Y = Xm
print(Xm_Y.shape)
## FPN结构添加 FPN(Xm, filter_num, kennel_size, UP_name)
Xm = FPN(Xm, filter_num=10, UP_size=(1, 2), UP_name="p5_up")
Xm_Conv1 = tf.keras.layers.Conv2D(10, (2, 2), padding="SAME", name="p5_cov")(Xm)
Xm = Xm+Xm_R3
Xm = FPN(Xm, filter_num=10, UP_size=(1, 2), UP_name="p4_up")
print(Xm.shape)
Xm_Conv2 = tf.keras.layers.Conv2D(10, (2, 2), padding="SAME", name="p4_cov")(Xm)
print(Xm.shape)
print(Xm_R2.shape)
Xm = Xm+Xm_R2
Xm = FPN(Xm, filter_num=10, UP_size=(1, 2), UP_name="p3_up")
Xm_Conv3 = tf.keras.layers.Conv2D(10, (2, 2), padding="SAME", name="p3_cov")(Xm)
Xm = Xm+Xm_R1
Xm = FPN(Xm, filter_num=10, UP_size=(1, 2), UP_name="p2_up")
Xm_Conv4 = tf.keras.layers.Conv2D(10, (2, 2), padding="SAME", name="p2_cov")(Xm)
## PAN结构的搭建,不仅是下采样，对分别从高维语义特征与低维语义特征进行分析处理print(Xm.shape),print(Xm_Conv4.shape)
Xm = PAN(Xm, filter_num=10, kennel_size=(1, 2), Conv_name="PAN_Conv1", pool_size=(1, 1), strides=(1, 1), upsample=True)
Xm = Xm+Xm_Conv4
Xm = PAN(Xm, filter_num=10, kennel_size=(1, 2), Conv_name="PAN_Conv2", pool_size=(1, 2), strides=(1, 2), upsample=True)
Xm = Xm+Xm_Conv3
Xm = PAN(Xm, filter_num=10, kennel_size=(1, 2), Conv_name="PAN_Conv3", pool_size=(1, 2), strides=(1, 2), upsample=True)
Xm = Xm+Xm_Conv2
Xm = PAN(Xm, filter_num=10, kennel_size=(1, 2), Conv_name="PAN_Conv4", pool_size=(1, 2), strides=(1, 2), upsample=False)
Xm = Xm+Xm_Conv1
Xm = PAN(Xm, filter_num=10, kennel_size=(1, 2), Conv_name="PAN_Conv5", pool_size=(1, 2), strides=(1, 2), upsample=False)
Xm = Xm+Xm_Y
print(Xm.shape)

#Full Con 1
Xm = Flatten(data_format=data_format)(Xm)
# Xm = Dense(32, activation='selu', kernel_initializer='glorot_normal', name="dense1")(Xm)
Xm = AlphaDropout(0.3)(Xm)# 丢弃率
#Full Con 2
Xm = Dense(len(classes), kernel_initializer='glorot_normal', name="dense2")(Xm)
#SoftMax
Xm = Activation('softmax')(Xm)
#Create Model
model = Model.Model(inputs=Xm_input, outputs=Xm)
adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam)
model.summary()

"""训练模型"""

print(tf.test.gpu_device_name())
filepath = 'C:/pycharm/ResNet-for-Radio-Recognition/own_models/ResNet_Model_72w.h5'

model.compile(loss='categorical_crossentropy',
               optimizer=adam,
               metrics=['categorical_accuracy'])

time_callback = TimeHistory()
history = model.fit(X_train,
    Y_train,
    batch_size=128,
    epochs=epochs_num,#100
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto'), time_callback
    ])
scores = model.evaluate(X_train, Y_train)
model.load_weights(filepath)

with open('model_struct_1_R.json', 'w') as f:
    json.dump(model.to_json(), f)
model.save_weights('model_weights_1_R.h5')

loss_list = history.history['loss']
val_loss_list = history.history['val_loss']
plt.plot(range(len(loss_list)), loss_list, linewidth=2)
plt.plot(range(len(loss_list)), val_loss_list, linewidth=2)
# plt.title("Loss", fontsize=18)
plt.ylabel("loss", fontsize=15)
plt.xlabel("Number of iterations", fontsize=15)
plt.legend(["training loss", "test loss"], loc="upper right")
plt.savefig("loss and val_loss.png", format='png', dpi=300, bbox_inches = 'tight')
plt.show()

def plot_confusion_matrix(cm, title, cmap=plt.cm.gray_r, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title, fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig(title, format='png', dpi=300, bbox_inches = 'tight')
    plt.show()
# Plot.confusion matrix
batch_size = 1024
test_Y_hat = model.predict(X_test, batch_size=1024)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
plot_confusion_matrix(confnorm, title='Confusion matrix.png', labels=classes)

for i in range(len(confnorm)):
    print(classes[i], confnorm[i, i])

acc = {}
Z_test = Z_test.reshape((len(Z_test)))
SNRs = np.unique(Z_test)
for snr in SNRs:
    X_test_snr = X_test[Z_test == snr]
    Y_test_snr = Y_test[Z_test == snr]

    pre_Y_test = model.predict(X_test_snr)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, X_test_snr.shape[0]):  # 该信噪比下测试数据量
        j = list(Y_test_snr[i, :]).index(1)  # 正确类别下标
        j = classes.index(classes[j])
        k = int(np.argmax(pre_Y_test[i, :]))  # 预测类别下标
        k = classes.index(classes[k])
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    # plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d).png" % (snr))
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy %s: " % snr, cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)

# 图形绘制
plt.plot(acc.keys(), acc.values(), 'o-', color='blue', linewidth=2)
plt.ylabel('Recognition accuracy', fontsize=12)
plt.xlabel('SNR', fontsize=12)
plt.savefig("model_acc.png", format='png', dpi=300, bbox_inches='tight')
res = np.vstack((acc.keys(), acc.values()))
savefile_name = 'Layer_num=' + str(Floor) + '.txt'
np.savetxt(savefile_name, res, delimiter=',', fmt='%s')
plt.show()

#存储训练与测试损失值与精度
pd.DataFrame.from_dict(history.history).to_csv("Loss_ACC.csv", float_format="%.5f", index=False)

plt.plot(history.history['categorical_accuracy'], 'o-', color='blue', linewidth=2)
plt.xlim((0, epochs_num))
plt.ylim((0, 1))
plt.ylabel("Recognition accuracy", fontsize=15)
plt.xlabel("Number of iterations", fontsize=15)
plt.savefig("model_acc.png", format='png', dpi=300, bbox_inches = 'tight')
X = list(range(0, epochs_num))# 迭代次数
res1 = np.vstack((X, history.history['categorical_accuracy']))
np.savetxt('epochs_acc.txt', res1)
plt.show()
# 运行时间打印
print(time_callback.times)
print(time_callback.totaltime)
# # 计算每次迭代的损失值与识别率
pd.DataFrame.from_dict(history.history).to_csv("Loss_ACC.csv", float_format="%.5f", index=False)