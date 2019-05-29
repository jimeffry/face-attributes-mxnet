# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/5/24 10:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face 
####################################################
from mxnet.gluon.model_zoo import vision
import mxnet as mx
from mxnet import nd
import os 
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


def normal(image):
    image = image/255
    normalized = mx.image.color_normalize(image,
                mean=mx.nd.array([0.485, 0.456, 0.406]),
                std=mx.nd.array([0.229, 0.224, 0.225]))

def get_pretrained(ctx):
    '''
    #mobilenetv2_1.0
    '''
    assert cfgs.NET_NAME in cfgs.NetList, 'please select a net in netlist'
    model_net = vision.get_model(cfgs.NET_NAME,pretrained=True,ctx=ctx,root='../../models')
    '''
    if cfgs.NET_NAME == 'mobilenet_v2_1_0' :
        #model_net = vision.mobilenet_v2_1_0(pretrained=True,ctx=ctx)
        model_net = vision.get_mobilenet_v2(1.0,pretrained=True,ctx=ctx,root='../../models')
    elif cfgs.NET_NAME == 'resnet18_v1':
        model_net = vision.resnet18_v1(pretrained=True)
    elif cfgs.NET_NAME == 'MobileNet':
        model_net = vision.get_mobilenet(1.0,pretrained=True)
    elif cfgs.NET_NAME == 'resnet50_v1':
        model_net = vision.resnet50_v1()
    '''
    return model_net

def get_symbol(ctx):
    net = get_pretrained(ctx=ctx)
    #net.hybridize()
    #net.summary(nd.zeros((1, 3, 224, 224),ctx=mx.cpu(0)))
    #print(net)
    return net

if __name__ == '__main__':
    get_symbol(mx.cpu(0))