# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/05/30 19:09
#project: Face attribute
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face attribute 
####################################################
import mxnet as mx
import numpy as np 
import time 
import cv2
import argparse
import os 
import sys
from tqdm import tqdm
from get_model import Face_Anti_Spoof
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--out-file',type=str,dest='out_file',default='None',\
                        help="the file output path")
    parser.add_argument('--data-file',type=str,dest='data_file',default='None',\
                        help="the file input path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=50,\
                        help="scale img size")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--img-path2',type=str,dest='img_path2',default="test2.jpg",\
                        help="scale img size")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./base_dir",\
                        help="images saved dir")
    parser.add_argument('--caffemodel',type=str,dest='m_path',default="../models/deploy.caffemodel",\
                        help="caffe model path")
    parser.add_argument('--mx-model',type=str,dest='mx_model',default="../../models/",\
                        help="models saved dir")
    parser.add_argument('--gpu', default=None, type=str,help='which gpu to run')
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./saved_dir",\
                        help="images saved dir")
    parser.add_argument('--failed-dir',type=str,dest='failed_dir',default="./failed_dir",\
                        help="fpr saved dir")
    parser.add_argument('--load-epoch', default=0,dest='load_epoch', type=int,help='saved epoch num')
    parser.add_argument('--cmd-type', default="dbtest",dest='cmd_type', type=str,\
                        help="which code to run: videotest,imgtest ")
    return parser.parse_args()

def test_img(args):
    '''
    '''
    img_path1 = args.img_path1
    model_dir = args.mx_model
    model_dir = os.path.join(model_dir,cfgs.DATASET_NAME)
    model_path = os.path.join(model_dir,cfgs.MODEL_PREFIX)
    Model = Face_Anti_Spoof(model_path,args.load_epoch,cfgs.IMG_SIZE,args.gpu,layer='fc')
    img_data1 = cv2.imread(img_path1)
    if img_data1 is None:
        print('img is none')
        return None
    #fram_h,fram_w = img_data1.shape[:2]
    img_data1 = np.expand_dims(img_data1,0)
    tmp,pred_id = Model.inference(img_data1)
    print("pred",tmp)
    score_label = "{}".format(cfgs.DATA_NAME[pred_id[0]])
    cv2.putText(img_data1,score_label,(int(20),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("video",img_data1)
    cv2.waitKey(0)

def display(img,id):
    #fram_h,fram_w = img.shape[:2]
    score_label = "{}".format(cfgs.FaceProperty[id])
    cv2.putText(img,score_label,(int(20),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("video",img)
    cv2.waitKey(0)

def evalue(args):
    '''
    calculate the tpr and fpr for all classes
    real positive: tp+fn
    real negnitive: fp+tn
    R = tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    P = tp/(tp+fp)
    '''
    file_in = args.file_in
    result_out = args.out_file
    img_dir = args.base_dir
    model_dir = args.mx_model
    model_dir = os.path.join(model_dir,cfgs.DATASET_NAME)
    model_path = os.path.join(model_dir,cfgs.MODEL_PREFIX)
    if args.gpu is None:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(0)
    Model = Face_Anti_Spoof(model_path,args.load_epoch,cfgs.IMG_SIZE,ctx=ctx)
    if file_in is None:
        print("input file is None",file_in)
        return None
    file_rd = open(file_in,'r')
    file_wr = open(result_out,'w')
    file_cnts = file_rd.readlines()
    total_num = len(file_cnts)
    statistics_dic = dict()
    for name in cfgs.FaceProperty:
        statistics_dic[name+'_tp'] = 0
        statistics_dic[name+'_tn'] = 0
        statistics_dic[name+'_tpfn'] = 0
        statistics_dic[name+'_fptn'] = 0
        #statistics_dic[name] = 0
    for i in tqdm(range(total_num)):
        item_cnt = file_cnts[i]
        item_spl = item_cnt.strip().split(',')
        img_path = item_spl[0]
        real_label = item_spl[1:]
        img_path = os.path.join(img_dir,img_path)
        img_data = cv2.imread(img_path)
        if img_data is None:
            print('img is none',img_path)
            continue
        img_data = np.expand_dims(img_data,0)
        probility,pred_id = Model.inference(img_data)
        pred_ids = pred_id[0]
        for idx in range(cfgs.CLS_NUM):
            pred_cls_id = pred_ids[idx]
            real_cls_id = int(real_label[idx])
            pred_name = cfgs.FaceProperty[idx]
            #real_name = cfgs.FaceProperty[int(real_label)]
            real_name = cfgs.FaceProperty[idx]
            if real_cls_id:
                statistics_dic[real_name+'_tpfn'] +=1
            else:
                statistics_dic[real_name+'_fptn'] +=1          
            if int(pred_cls_id) == int(real_cls_id)==1:
                statistics_dic[pred_name+'_tp'] +=1
            elif int(pred_cls_id)==int(real_cls_id)==0:
                statistics_dic[pred_name+'_tn'] +=1
            if cfgs.ShowImg:
                display(img_data,idx)
    for key_name in cfgs.FaceProperty:
        tp_fn = statistics_dic[key_name+'_tpfn']
        tp = statistics_dic[key_name+'_tp']
        tn = statistics_dic[key_name+'_tn']
        fp_tn = statistics_dic[key_name+'_fptn']
        fp = fp_tn - tn
        tpr = float(tp) / tp_fn if tp_fn else 0.0
        fpr = float(fp) / fp_tn if fp_tn else 0.0
        precision = float(tp) / (tp+fp) if tp+fp else 0.0
        statistics_dic[key_name+'_tpr'] = tpr
        statistics_dic[key_name+'_fpr'] = fpr
        statistics_dic[key_name+'_P'] = precision
        #file_wr.write('>>> {} result is: tp_fn-{} | fp_tn-{} | tp-{} | fp-{}\n'.format(key_name,\
         #               tp_fn,fp_tn,tp,fp))
        #file_wr.write('\t tpr:{:.4f} | fpr:{:.4f} | Precision:{:.4f}\n'.format(tpr,fpr,precision))
        file_wr.write("{}\t{}\t{}\t{}\n".format(key_name,tpr,fpr,precision))
    file_rd.close()
    file_wr.close()

if __name__ == '__main__':
    parms = args()
    cmd_type = parms.cmd_type
    if cmd_type in 'imgtest':
        test_img(parms)
    elif cmd_type in 'evalue':
        evalue(parms)
    else:
        print('Please input right cmd')