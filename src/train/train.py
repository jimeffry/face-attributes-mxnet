# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/5/27 10:09
#project: face attributes
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face 
####################################################
import os
import sys
import numpy as np 
import argparse
import time
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx 
from mxnet import nd
from train_model import load_model,get_layer_output,load_parms,add_layer,graph,fit,get_pretrained_layer
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../symbols'))
from gloun_net import get_symbol
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_data'))
from rec2img import DataIterLoader,FaceImageIter

def parms():
    parser = argparse.ArgumentParser(description='Face-attribute training')
    parser.add_argument('--load-num',dest='load_num',type=str,default=None,help='ckpt num')
    parser.add_argument('--save-weight-period',dest='save_weight_period',type=int,default=5,\
                        help='the period to save')
    parser.add_argument('--epochs',type=int,default=20000,help='train epoch nums')
    parser.add_argument('--batch-size',dest='batch_size',type=int,default=32,\
                        help='train batch size')
    parser.add_argument('--model-dir',dest='model_dir',type=str,default='../../models/',\
                        help='path saved models')
    parser.add_argument('--log-path',dest='log_path',type=str,default='../../log',\
                        help='path saved logs')
    parser.add_argument('--gpu-list',dest='gpu_list',type=str,default=None,\
                        help='train on gpu num')
    parser.add_argument('--data-record-dir',dest='data_record_dir',type=str,\
                        default='../../data/',help='mxnet data record')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
    return parser.parse_args()


def train(args):
    load_num = int(args.load_num)
    data_record_dir = args.data_record_dir
    data_record_dir = os.path.join(data_record_dir,cfgs.DATASET_NAME)
    log_dir = args.log_path
    gpu_list = args.gpu_list
    batch_size = args.batch_size
    #*****************************************************************set log 
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(log_dir,time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'))
    fh.setLevel(logging.DEBUG)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    # logger.addHandler(ch)
    #*********************************************************get train model 
    if gpu_list is None:
        devs = mx.cpu(0)
    else:
        #devs = [mx.gpu(int(i)) for i in range(len(gpu_list.split(',')))]
        devs = mx.gpu(0)
    #logging.info("use gpu list: ",devs)
    '''
    sym,arg_param,aux_param = load_model(load_num)
    net,new_arg,new_aux = get_layer_output(sym,arg_param,aux_param,'flatten')
    net_load = load_parms(net,new_arg,new_aux,devs)
    '''
    #mobilenetv20_features_pool0_fwd
    net_train = get_symbol(devs)
    net_train = get_pretrained_layer(net_train,'resnetv10_pool1_fwd')
    sigmoid_layer = add_layer(devs)
    net_train = graph(net_train,sigmoid_layer) 
    net_train.hybridize()
    #net_train.summary(nd.zeros((1, 3, 224, 224),ctx=mx.cpu(0)))
    #*******************************************************************load data
    train_rec_path = os.path.join(data_record_dir,'train.rec')
    val_rec_path = os.path.join(data_record_dir,'test.rec')
    train_dataiter = FaceImageIter(
          batch_size           = batch_size,
          data_shape           = (3,112,112),
          path_imgrec          = train_rec_path,
          shuffle              = True,
          cutoff               = 0,
      )
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    train_loader = DataIterLoader(train_dataiter)
    val_dataiter = FaceImageIter(
          batch_size           = batch_size,
          data_shape           = (3,112,112),
          path_imgrec          = val_rec_path,
          shuffle              = True,
          cutoff               = 0,
      )
    val_dataiter = mx.io.PrefetchingIter(val_dataiter)
    val_loader = DataIterLoader(val_dataiter)
    #******************************************************************train
    fit(net_train,train_loader,val_loader,\
        ctx=devs,
        epoch=args.epochs,
        save_epoch=args.save_weight_period,
        load_epoch=load_num,
        learning_rate=args.lr,
        batch_size=batch_size,
        model_dir=args.model_dir)


if __name__=='__main__':
    args = parms()
    train(args)