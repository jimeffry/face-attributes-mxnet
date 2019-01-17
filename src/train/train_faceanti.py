# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/1/16 10:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face 
####################################################
import os
import sys 
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import time
import mxnet as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '../common'))
import find_mxnet, fit,data
from util import download_file
sys.path.append(os.path.join(os.path.dirname(__file__), '../prepare_data'))
from rec2img import FaceImageIter

thisdir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train FaceAnti",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    #data.add_data_args(parser)
    #data.add_data_aug_args(parser)
    # use a large aug level
    #data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network          = 'mobilenetv2',
        multiplier       = 1.0, 
        model_prefix     = '../../models/FaceAnti/mobilenetv2-1_0',
        # data
        train_rec        = '../../data/FaceAnti_train.rec',
        val_rec          = '../../data/FaceAnti_test.rec',
        rand_mirror      = 1,
        rgb_mean         = 127.5,
        num_classes      = 4,
        num_examples     = 6013,
        # train
        num_epochs       = 480, # default=480 epochs
        lr               = 0.045, # default=0.045 
        lr_factor        = 0.98, # default=0.98
        lr_step_epochs   = ','.join([str(i) for i in range(1,480)]),
        wd               = 0.00004, 
        dtype            = 'float32', 
        batch_size       = 64,
        gpus             = '0',
        optimizer        = 'sgd',
        monitor          = 20, 
        load_epoch       = 5, # default=None
        top_k            = 0,
        display          = 100,
        savestep         = 1000,
    )
    args = parser.parse_args()
    from pprint import pprint
    #pprint(vars(args))
    train_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = (3,224,224),
          path_imgrec          = args.train_rec,
          shuffle              = True,
          rand_mirror          = args.rand_mirror,
          mean                 = args.rgb_mean,
          cutoff               = 0,
      )
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    val_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = (3,224,224),
          path_imgrec          = args.val_rec,
          shuffle              = True,
          rand_mirror          = args.rand_mirror,
          mean                 = args.rgb_mean,
          cutoff               = 0,
      )
    val_dataiter = mx.io.PrefetchingIter(val_dataiter)
    # load network
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(num_classes=args.num_classes, multiplier=args.multiplier)
    # print(sym.get_internals()['mobilenetv20_features_conv0_weight'].attr_dict()['mobilenetv20_features_conv0_weight']['__shape__'])
    # exit()
    # set up logger
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join('../../log',time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'))
    fh.setLevel(logging.DEBUG)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    # logger.addHandler(ch)
    # train
    fit.fit(args, sym,train_dataiter,val_dataiter, logger)
