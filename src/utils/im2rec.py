#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from __future__ import print_function
import os
import sys

#curr_path = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

def list_image(root, recursive, exts,label_id=0):
    """Traverses the root of directory that contains images and
    generates image list iterator.
    Parameters
    ----------
    root: string
    recursive: bool
    exts: string
    Returns
    -------
    image iterator that contains all the image under the specified path
    """

    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            #print("len",len(files))
            #print("dir",dirs)
            path_name = path.strip().split('/')[-1]
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    #if path not in cat:
                        #cat[path] = len(cat)
                    #print('path',dirs)
                    if path_name =='mobilephone':
                        label_id = 1
                    elif path_name == 'TV':
                        label_id = 2
                    elif path_name == 'telecontroller':
                        label_id = 3
                    else:
                        label_id = 0
                    img = cv2.imread(fpath)
                    if img is None:
                        print("img none",fpath)
                        continue
                    yield (i, os.path.relpath(fpath, root), label_id) #cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), label_id)
                i += 1

def write_list(path_out, image_list):
    """Hepler function to write image list into the file.
    The format is as below,
    integer_image_index \t float_label_index \t path_to_image
    Note that the blank between number and tab is only used for readability.
    Parameters
    ----------
    path_out: string
    image_list: list
    """
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%d\t' % j
            line += '%s\n' % item[1]
            fout.write(line)

def make_list(args):
    """Generates .lst file.
    Parameters
    ----------
    args: object that contains all the arguments
    """
    image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        print("shuffle list")
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + args.chunks - 1) // args.chunks
    for i in range(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * args.train_ratio)
        sep_test = int(chunk_size * args.test_ratio)
        if args.train_ratio == 1.0:
            write_list(args.prefix + str_chunk + '.lst', chunk)
        else:
            if args.test_ratio:
                write_list(args.prefix + str_chunk + '_test.lst', chunk[:sep_test])
            if args.train_ratio + args.test_ratio < 1.0:
                write_list(args.prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])
            write_list(args.prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])
    print("make list over")



def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('--prefix',type=str, help='prefix of input/output lst and rec files.')
    parser.add_argument('--root', type=str,help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', type=bool,default=False, #action='store_true',
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', type=bool,default=False, #action='store_true',
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    cgroup.add_argument('--no-shuffle', dest='shuffle', type=bool,default=True,#action='store_true',
                        help='If this is passed, \
        im2rec will not randomize the image order in <prefix>.lst')
    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--pass-through', dest='pass_through',type=bool,default=False,#action='store_true',
                        help='whether to skip transformation and save image as is')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center-crop',dest='center_crop',type=bool,default=True,# action='store_true',
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--pack-label', action='store_true',
        help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args

if __name__ == '__main__':
    args = parse_args()
    # if the '--list' is used, it generates .lst file
    if int(args.list):
        print('make list',args.list)
        make_list(args)
    # otherwise read .lst file to generates .rec file
    else:
        print("Please input list is True")
