# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/04/23 10:09
#project: Face recognize
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  test caffe model and figure confusionMatrix
####################################################
# coding: utf-8
# YuanYang
import math
import cv2
import numpy as np

def nms(rectangles,threshold,mode='Union'):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    #I[-1] have hightest prob score, I[0:-1]->others
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) 
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    #result_rectangle = boxes[pick].tolist()
    return pick

def adjust_input(in_data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input
    Parameters:
        in_data: numpy array of shape (h, w, c)
    Returns:
        out_data: numpy array of shape (1, c, h, w)
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data
    out_data = out_data.transpose((2,0,1))
    out_data = np.expand_dims(out_data, 0)
    out_data = (out_data - 127.5)*0.0078125
    return out_data

def generate_bbox(feature_map, reg, scale, threshold):
     """
         generate bbox from feature map
     Parameters:
         map: numpy array , n x m x 1
             detect score for each position
         reg: numpy array , n x m x 4
             bbox
         scale: float number
             scale of this detection
         threshold: float number
             detect threshold
     Returns:bbox array
     """
     stride = 2
     cellsize = 12
     t_index = np.where(feature_map>threshold)
     # find nothing
     if t_index[0].size == 0:
         return []
     dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]
     reg = np.array([dx1, dy1, dx2, dy2])
     score = feature_map[t_index[0], t_index[1]]
     boundingbox = np.vstack([np.round((stride*t_index[1]+1)/scale),
                              np.round((stride*t_index[0]+1)/scale),
                              np.round((stride*t_index[1]+1+cellsize)/scale),
                              np.round((stride*t_index[0]+1+cellsize)/scale),
                              score,
                              reg])

     return boundingbox.T


def detect_first_stage(img, net, scale, threshold):
    """
        run PNet for first stage
    Parameters:
        img: numpy array, bgr order
        scale: float number, how much should the input image scale
        net: PNet
    Returns:
        total_boxes : bboxes
    """
    height, width, _ = img.shape
    hs = int(math.ceil(height * scale))
    ws = int(math.ceil(width * scale))
    img = cv2.resize(img, (ws,hs))
    # adjust for the network input
    img = adjust_input(img)
    output = net.predict(img)
    #print("output,",np.shape(output[1]))
    boxes = generate_bbox(output[1][0,1,:,:], output[0], scale, threshold)
    if len(boxes) == 0:
        #print('in pnet',scale)
        return []
    # nms
    pick = nms(boxes[:,0:5], 0.5, mode='Union')
    boxes = boxes[pick]
    return boxes