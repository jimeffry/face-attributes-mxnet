# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
import string
#curr_path = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback
#from builtins import range
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfgs

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

def parse_lst_line(line):
  vec = line.strip().split(",")
  #vec = line.strip().split(" ")
  assert len(vec)==22
  pack_type = 0 # pack:1 pack_img:0
  image_path = vec[0]
  #print('label',vec[1])
  #label = string.atoi(vec[1])
  #bbox = None
  #landmark = None
  label = []
  #print(vec)
  if len(vec)>3:
    '''
    bbox = np.zeros( (4,), dtype=np.int32)
    for i in xrange(3,7):
      bbox[i-3] = int(vec[i])
    landmark = None
    if len(vec)>7:
      _l = []
      for i in xrange(7,17):
        _l.append(float(vec[i]))
      landmark = np.array(_l).reshape( (2,5) ).T
    '''
    for i in range(1,22):
        label.append(int(vec[i]))
  return image_path, label, pack_type

def read_list(path_in):
    with open(path_in) as fin:
        last = [-1, -1]
        _id = 1
        while True:
            line = fin.readline()
            if not line:
                break
            item = edict()
            item.flag = 0
            item.image_path, item.label, item.pack_type = parse_lst_line(line)
            item.id = _id
            #item.label = [label, item.pack_type]

            yield item
            _id+=1
        #record the dataset info: image numbers and pack-type
        item = edict()
        item.flag = 2
        item.id = 0
        item.label = [float(_id),item.pack_type]
        yield item


def image_encode(args, i, item, q_out):
    oitem = [item.id]
    #print('flag', item.flag)
    if item.flag==0:
      #fullpath = item.image_path
      fullpath = os.path.join(args.root, item.image_path)
      header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
      #print('write', item.flag, item.id, item.label)
      #print("img_encode: ",fullpath)
      if item.pack_type:
        with open(fullpath, 'rb') as fin:
            img = fin.read()
            if cfgs.img_norm:
                img = (img-127.5)*0.0078125
        s = mx.recordio.pack(header, img)
        q_out.put((i, s, oitem))
      else:
        img = cv2.imread(fullpath, args.color)
        if img is None:
            print('imread read blank (None) image for file: %s' % fullpath)
            q_out.put((i, None, oitem))
            return
        img_format = '.' + fullpath.split('.')[-1]
        if args.center_crop:
            #print("make center crop")
            if img.shape[0] > img.shape[1]:
                margin = (img.shape[0] - img.shape[1]) // 2
                img = img[margin:margin + img.shape[1], :,:]
            else:
                margin = (img.shape[1] - img.shape[0]) // 2
                img = img[:, margin:margin + img.shape[0],:]
        if args.resize:
            #print("make resize")
            img = cv2.resize(img, (args.resize,args.resize))
        if cfgs.img_norm:
            img = (img-127.5)*0.0078125
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=img_format)
        q_out.put((i, s, oitem))
    else: 
      header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
      #print('write', item.flag, item.id, item.label)
      s = mx.recordio.pack(header, '')
      q_out.put((i, s, oitem))


def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)

def write_worker(q_out, fname, working_dir):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                #print('write idx', item[0])
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('--prefix',type=str, help='prefix of input/output lst and rec files.')
    parser.add_argument('--root',type=str, help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', type=bool, default=False,
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    cgroup.add_argument('--center-crop',dest='center_crop',type=bool,default=True,# action='store_true',
                        help='specify whether to crop the center image to make it rectangular.')
    cgroup.add_argument('--shuffle', type=bool, default=True, help='If this is set as True, \
        im2rec will randomize the image order in <prefix>.lst')

    cgroup = parser.add_argument_group('Options for creating database')
    cgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    cgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    cgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    cgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    cgroup.add_argument('--pack-label', type=bool, default=False,
        help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    #args.root = os.path.abspath(args.root)
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.list:
        pass
        #make_list(args)
        print("please set false")
    else:
        if os.path.isdir(args.prefix):
            working_dir = args.prefix
        else:
            working_dir = os.path.dirname(args.prefix)
        files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                    if os.path.isfile(os.path.join(working_dir, fname))]
        count = 0
        print("work dir: ",files)
        for fname in files:
            if  fname.endswith('.lst'):
                print('Creating .rec file from', fname, 'in', working_dir)
                count += 1
                image_list = read_list(fname)
                # -- write_record -- #
                if args.num_thread > 1 and multiprocessing is not None:
                    q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                    q_out = multiprocessing.Queue(1024)
                    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                                    for i in range(args.num_thread)]
                    for p in read_process:
                        p.start()
                    write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
                    write_process.start()

                    for i, item in enumerate(image_list):
                        q_in[i % len(q_in)].put((i, item))
                    for q in q_in:
                        q.put(None)
                    for p in read_process:
                        p.join()

                    q_out.put(None)
                    write_process.join()
                else:
                    print('multiprocessing not available, fall back to single threaded encoding')
                    try:
                        import Queue as queue
                    except ImportError:
                        import queue
                    q_out = queue.Queue()
                    fname = os.path.basename(fname)
                    fname_rec = os.path.splitext(fname)[0] + '.rec'
                    fname_idx = os.path.splitext(fname)[0] + '.idx'
                    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                           os.path.join(working_dir, fname_rec), 'w')
                    cnt = 0
                    pre_time = time.time()
                    for i, item in enumerate(image_list):
                        if item.id ==0:
                            label_cnt = item.label[0]
                        image_encode(args, i, item, q_out)
                        if q_out.empty():
                            continue
                        _, s, item = q_out.get()
                        if s is None:
                            continue
                        #header, _ = mx.recordio.unpack(s)
                        #print('write header label', header.label)
                        #header, img = mx.recordio.unpack_img(s)
                        #img.astype(np.uint8)
                        #cv2.imshow('img',img)
                        #cv2.waitKey(0)
                        record.write_idx(item[0], s)
                        if cnt % 1000 == 0:
                            cur_time = time.time()
                            print('time:', cur_time - pre_time, ' count:', cnt)
                            pre_time = cur_time
                        cnt += 1
                    print("over ",cnt)
                    print("ids ",label_cnt)
        if not count:
            print('Did not find and list file with prefix %s'%args.prefix)

