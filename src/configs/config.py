# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/20 17:09
#project: face anti spoofing
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face anti spoofing
####################################################
from easydict import EasyDict 

cfgs = EasyDict()

#------------------------------------------ convert data to tfrecofr config
cfgs.BIN_DATA = 0 # whether read image data from binary
cfgs.CLS_NUM = 21 #inlcude background:0, mobile:1  tv:2 remote-control:3
cfgs.IMG_NORM = 0
cfgs.FaceProperty = ['No_Beard','Mustache','Goatee','5_o_Clock_Shadow','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Bangs','Bald', \
        'Male','Wearing_Hat','Wearing_Earrings','Wearing_Necklace','Wearing_Necktie',\
        'Eyeglasses','Young','Smiling','Arched_Eyebrows','Bushy_Eyebrows','Blurry']
# ---------------------------------------- System_config
cfgs.NET_NAME = 'mobilenetv2_1.0'#'resnet100'  # 'mobilenetv2' 'resnet50' 'lenet5',mobilenetv2_1.0
cfgs.NetList = ['mobilenetv2_1.0','MobileNet','mobilenet1.0','MobileNetV2','resnet18_v1','resnet50_v1','resnet50_v2','resnet101_v1','alexnet']
cfgs.SHOW_TRAIN_INFO_INTE = 100
cfgs.SMRY_ITER = 500
cfgs.DATASET_NAME = 'CelebA' #'Mobile' 'Prison' FaceAnti Fruit
cfgs.DATASET_LIST = ['Prison', 'WiderFace','Mobile','FaceAnti','Fruit'] 
cfgs.DATA_NAME = ['bg','mobilephone','monitor','telecontroller']
cfgs.DATA_NUM = 201600
# ------------------------------------------ Train config
cfgs.RD_MULT = 0
cfgs.MODEL_PREFIX = 'mobilenetv2-2' #'mobilenetv2-2' #'mobilenetv2-1'
cfgs.IMG_SIZE = [112,112]
cfgs.BN_USE = True 
cfgs.WEIGHT_DECAY = 1e-5
cfgs.MOMENTUM = 0.9
cfgs.LR = [0.01,0.001,0.0005,0.0001,0.00001]
cfgs.DECAY_STEP = [10,20,35,50]
# -------------------------------------------- Data_preprocess_config 
cfgs.PIXEL_MEAN = [127.5,127.5,127.5] #[123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
cfgs.PIXEL_NORM = 128.0
cfgs.IMG_LIMITATE = 0
cfgs.IMG_SHORT_SIDE_LEN = 112
cfgs.IMG_MAX_LENGTH = 112
# -------------------------------------------- test model
cfgs.ShowImg = 0
cfgs.mx_version = 0
cfgs.debug = 0
cfgs.display_model = 0
cfgs.batch_use = 0
cfgs.model_resave = 0
cfgs.time = 0
cfgs.x_y = 1
cfgs.box_widen = 1