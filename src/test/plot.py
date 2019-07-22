# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/05/31 19:09
#project: Face attribute
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face attribute 
####################################################
import numpy as np 
import time 
import cv2
import argparse
import os 
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--out-file',type=str,dest='out_file',default='None',\
                        help="the file output path")
    parser.add_argument('--data-file',type=str,dest='data_file',default='None',\
                        help="the file input path")
    parser.add_argument('--base-name',type=str,dest='base_name',default='score',\
                        help="scale img size")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--img-path2',type=str,dest='img_path2',default="test2.jpg",\
                        help="scale img size")
    parser.add_argument('--high',type=int,default=210000,\
                        help="the highest data num y sticks")
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./saved_dir",\
                        help="images saved dir")
    parser.add_argument('--unit',type=int,default=10000,\
                        help="y sticks")
    parser.add_argument('--cmd-type', default="dbtest",dest='cmd_type', type=str,\
                        help="which code to run: videotest,imgtest ")
    return parser.parse_args()

def autolabel(rects,ax,w, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / w, height),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')

def Grouped_bar_chart(bar1_data,bar2_data,x_labels,bar3_data=None,name='Scores',high=210000,unit=10000):
    #men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
    #women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)
    assert len(bar1_data) == len(bar2_data)
    ind = np.arange(len(bar1_data))  # the x locations for the groups
    if not isinstance(bar1_data,tuple):
        bar1_data = tuple(bar1_data)
    if not isinstance(bar2_data,tuple):
        bar2_data = tuple(bar2_data)    
    if not isinstance(x_labels,tuple):
        x_labels = tuple(x_labels)
    fig = plt.figure(num=0,figsize=(20,10))
    ax = fig.add_subplot(111)
    #fig, ax = plt.subplots()
    if bar3_data is None:
        width = 0.2  # the width of the bars
        rects1 = ax.bar(ind - width/2, bar1_data, width,  label='positive')
        rects2 = ax.bar(ind + width/2, bar2_data, width, label='negtive')
    else:
        if not isinstance(bar3_data,tuple):
            bar3_data = tuple(bar3_data)
        width = 0.2  # the width of the bars
        rects1 = ax.bar(ind - width/2, bar1_data, width,  label='recall')
        rects2 = ax.bar(ind + width/2, bar2_data, width, label='precision')
        rects3 = ax.bar(ind + 3*width/2,bar3_data,width,label='fpr')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    #locs, labels = plt.xticks(ind, x_labels)  
    #for t in labels:  
     #   t.set_rotation(90)
    ax.set_ylabel(name)
    if bar3_data is None:
        ax.set_yticks(np.arange(0, int(high), int(unit)))
    else:
        ax.set_yticks(np.arange(0,100,10))
    ax.set_title('%s by group ' % name)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.legend()
    if bar3_data is None:
        autolabel(rects1, ax,2,"left")
        autolabel(rects2,ax,2, "right")
    else:
        autolabel(rects1,ax,2,'left')
        autolabel(rects2,ax,2,'center')
        autolabel(rects3,ax,2,'right')
    fig.tight_layout()
    plt.savefig("./output/%s.png" % name,format='png')
    plt.show()

def plot_bar(bar_data,x_labels,name):
    #menMeans = (20, 35, 30, 35, 27)
    #womenMeans = (25, 32, 34, 20, 25)
    #menStd = (2, 3, 4, 1, 2)
    #womenStd = (3, 5, 2, 3, 3)
    ind = np.arange(len(bar_data))  # the x locations for the groups
    if not isinstance(bar_data,tuple):
        bar_data = tuple(bar_data)    
    if not isinstance(x_labels,tuple):
        x_labels = tuple(x_labels)
    width = 0.35       # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, bar_data, width)
    #p2 = plt.bar(ind, womenMeans, width,
    #            bottom=menMeans, yerr=womenStd)
    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    plt.show()

def process_traindatas(file_in):
    f_r = open(file_in,'r')
    f_contents = f_r.readlines()
    positive_data = []
    negtive_data = []
    p_names = []
    n_names = []
    for idx,tmp in enumerate(f_contents):
        tmp_splits = tmp.strip().split(':')
        if idx %2 == 0:
            p_names.append(tmp_splits[0].strip()[:-2])
            positive_data.append(int(tmp_splits[-1]))
        else:
            n_names.append(tmp_splits[0].strip())
            negtive_data.append(int(tmp_splits[-1]))
    f_r.close()
    return positive_data,p_names,negtive_data,n_names

def process_roc(file_in):
    f_r = open(file_in,'r')
    f_ctx = f_r.readlines()
    label_name = []
    tpr_data = []
    fpr_data = []
    precision_data = []
    for tmp in f_ctx:
        tmp_splits = tmp.split('\t')
        label_name.append(tmp_splits[0].strip())
        tpr_data.append(np.round(float(tmp_splits[1])*100,1))
        fpr_data.append(np.round(float(tmp_splits[2])*100,1))
        precision_data.append(np.round(float(tmp_splits[3])*100,1))
    f_r.close()
    return tpr_data,fpr_data,precision_data,label_name

if __name__=='__main__':
    parm = args()
    file_in = parm.file_in
    base_name = parm.base_name
    high_s = parm.high
    unit_s = parm.unit
    if parm.cmd_type == 'plot2data':
        positive_data,p_names,negtive_data,n_names = process_traindatas(file_in)
        Grouped_bar_chart(negtive_data,positive_data,p_names,name=base_name,high=high_s,unit=unit_s)
    elif parm.cmd_type == 'plot3data':
        tpr_data,fpr_data,precision_data,label_name = process_roc(file_in)
        Grouped_bar_chart(tpr_data,precision_data,label_name,fpr_data,base_name)
