# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import subprocess
import os
import errno
import numpy as np 
import cv2
import argparse

def parms():
    parser = argparse.ArgumentParser(description='gen ')
    parser.add_argument('--img-dir',type=str,dest="img_dir",default='./',\
                        help='the directory should include 2 more Picturess')
    parser.add_argument('--file-in',type=str,dest="file_in",default="train.txt",\
                        help='img paths saved file')
    parser.add_argument('--save-dir',type=str,dest="save_dir",default='./',\
                        help='img saved dir')
    parser.add_argument('--base-id',type=int,dest="base_id",default=0,\
                        help='img id')
    parser.add_argument('--out-file',type=str,dest="out_file",default="train.txt",\
                        help='out img paths saved file')
    parser.add_argument('--cmd-type',type=str,dest="cmd_type",default="None",\
                        help='which code to run: gen_trainfile')
    parser.add_argument('--file2-in',type=str,dest="file2_in",default="train2.txt",\
                        help='label files')
    return parser.parse_args()

def download_file(url, local_fname=None, force_write=False):
    # requests is not default installed
    import requests
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    dir_name = os.path.dirname(local_fname)

    if dir_name != "":
        if not os.path.exists(dir_name):
            try: # try to create the directory if it doesn't exists
                os.makedirs(dir_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname

def get_gpus():
    """
    return a list of GPUs
    """
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return []
    return range(len([i for i in re.split('\n') if 'GPU' in i]))

def get_by_ratio(x,new_x,y):
    ratio = x / float(new_x)
    new_y = y / ratio
    return np.floor(new_y)

def Img_Pad(img,crop_size):
    '''
    img: input img data
    crop_size: [h,w]
    '''
    img_h,img_w = img.shape[:2]
    d_h,d_w = crop_size
    pad_l,pad_r,pad_u,pad_d = [0,0,0,0]
    if img_w > d_w or img_h > d_h :
        if img_h> img_w:
            new_h = d_h
            new_w = get_by_ratio(img_h,new_h,img_w)
            if new_w > d_w:
                new_w = d_w
                new_h = get_by_ratio(img_w,new_w,img_h)
                if new_h > d_h:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_u = np.round((d_h - new_h)/2.0)
                    pad_d = d_h - new_h - pad_u
            else:
                pad_l = np.round((d_w - new_w)/2.0)
                pad_r = d_w - new_w - pad_l
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
        else:
            new_w = d_w
            new_h = get_by_ratio(img_w,new_w,img_h)
            if new_h > d_h:
                new_h = d_h
                new_w = get_by_ratio(img_h,new_h,img_w)
                if new_w > d_w:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_l = np.round((d_w - new_w)/2.0)
                    pad_r = d_w - new_w - pad_l
            else:
                pad_u = np.round((d_h - new_h)/2.0)
                pad_d = d_h - new_h - pad_u
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
    elif img_w < d_w or img_h < d_h:
        if img_h < img_w:
            new_h = d_h
            new_w = get_by_ratio(img_h,new_h,img_w)
            if new_w > d_w:
                new_w = d_w
                new_h = get_by_ratio(img_w,new_w,img_h)
                if new_h > d_h:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_u = np.round((d_h - new_h)/2.0)
                    pad_d = d_h - new_h - pad_u
            else:
                pad_l = np.round((d_w - new_w)/2.0)
                pad_r = d_w - new_w - pad_l
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
        else:
            new_w = d_w
            new_h = get_by_ratio(img_w,new_w,img_h)
            #print("debug1",new_h,new_w)
            if new_h > d_h:
                new_h = d_h
                new_w = get_by_ratio(img_h,new_h,img_w)
                #print("debug2",new_h,new_w)
                if new_w > d_w:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_l = np.round((d_w - new_w)/2.0)
                    pad_r = d_w - new_w - pad_l
            else:
                pad_u = np.round((d_h - new_h)/2.0)
                pad_d = d_h - new_h - pad_u
            #print("up",new_h,new_w)
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
    elif img_w==d_w and img_h==d_h:
        img_out = img
    if not [pad_l,pad_r,pad_u,pad_d] == [0,0,0,0] :
        color = [255,255,255]
        #print("padding",[pad_l,pad_r,pad_u,pad_d])
        img_out = cv2.copyMakeBorder(img_out,top=int(pad_u),bottom=int(pad_d),left=int(pad_l),right=int(pad_r),\
                                    borderType=cv2.BORDER_CONSTANT,value=color) #BORDER_REPLICATE
    return img_out

def generate_train_label(file_in,fileout):
    '''
    file_in: input_label csv file
    fileout: ouput train file
    '''
    f_in = open(file_in,'rb')
    dict_keys = f_in.readline().strip().split('')
    f_in.close()
    f_in = open(file_in,'rb')
    print('read data dict_keys:',dict_keys)
    list_out = []
    for name in dict_keys:
        list_out.append([])
    data_dict = dict(zip(dict_keys,list_out))
    reader = csv.DictReader(f_in)
    print(len(reader))
    #for f_item in reader:
        #print(f_item['filename'])
    f_in.close()
    
if __name__ == '__main__':
    args = parms()
    file_in = args.file_in
    file_out = args.out_file
    cmd = args.cmd_type
    if cmd=='gen_trainfile':
        generate_train_label(file_in,file_out)
    else:
        print("please input cmd right")