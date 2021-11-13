# This repository is about the research on the enumeration method of the number of radio signal sources, and the network training and verification on the real measured data set has confirmed the effectiveness of the proposed MFFNet.
![MFFNet](https://user-images.githubusercontent.com/74703156/141611152-13298899-8cf4-4660-8417-526e64155f9c.jpg)

# Environments

python 3.8.6

PyCharm Community 2018.3.2

CUDA 11

# Requirements

h5py 2.10.0

numpy 1.19.5

tensorflow-gpu 2.4.0

matplotlib 3.3.3

pandas 1.2.4

# File description

(1)main_demo is a complete network program, including: residual, SEnet, FPN, PAN modules;

(2)test_data is a test data set with a slice length of 128;

(3)test_demo is a test program, where test_no_virtual.py is a test program, and plot_acc.py is a program for drawing recognition accuracy;

(4)weight_and_biase_of_Network is a folder for saving network architecture and weight. Among them, ResNet saves the architecture and weights of the trained Residual network, ResNet_SEnet saves the architecture and weights of the trained ResNet+SEnet, and ResNet_SEnet_FPN_PAN saves the architecture and weights of the trained ResNet+SEnet+FPN+PAN network;

(5)The network architecture file is named: model_struct_1_R.json, the network weight is named: model_weights_1_R.h5.

# Test steps
(1) Copy the model_struct_1_R.json and model_weights_1_R.h5 of the relevant network in weight_and_biase_of_Network to the test_demo folder;

(2) Open test_no_virtual.py in test_demo and modify the test data address in the program to the absolute address of test_data;

(3) Run test_no_virtual.py in test_demo to get the pred_confusion_mat_L128.npy file;

(4) Run plot_acc.py to get the recognition accuracy map L128.png and the recognition accuracy file acc.txt;

(5) Perform the above steps (1)-(3) on the relevant networks in turn to complete the test on all networks.

# Contact
Issues should be raised directly in the repository. For professional support requests please email Rong Fan at fanrong@cafuc.edu.cn.











