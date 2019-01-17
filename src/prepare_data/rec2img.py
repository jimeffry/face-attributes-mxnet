from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import multiprocessing

logger = logging.getLogger()

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--train-rec',dest='train_rec',type=str, help='the training data')
    data.add_argument('--val-rec',dest='val_rec',type=str, help='the validation data')
    data.add_argument('--rgb-mean',dest='rgb_mean', type=float, default=None,
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--image-shape', type=str,
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--cutoff', type=int, default=0,
                      help='data cutoff')
    data.add_argument('--rand-mirror',dest='rand_mirror', type=int, default=0,
                      help='if 1, then mirror data')
    data.add_argument('--num-classes',dest='num_classes', type=int, default=0,
                      help='class num')
    data.add_argument('--num-examples',dest='num_examples', type=int, default=1,
                      help='data num to train')
    return data

class FaceImageIter(io.DataIter):
    def __init__(self, batch_size, data_shape,
                 path_imgrec = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag==2:
                #print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                #assert(header.flag==1)
                self.imgidx = range(1, int(header.label[0]))
            else:
              #print("header flag ",header.flag)
                self.imgidx = list(self.imgrec.keys)
            if shuffle:
                self.seq = self.imgidx
                self.oseq = self.imgidx
                #print("init shutffle",len(self.seq))
            else:
                self.seq = None
        self.mean = mean
        if self.mean:
            self.mean = np.ones([1,1,3], dtype=np.float32)*self.mean
            self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        self.cutoff = cutoff
        self.provide_label = [(label_name, (batch_size,))]
        #print(self.provide_label[0][1])
        self.cur = 0
        self.nbatch = 0
        self.is_init = False


    def reset(self):
        """Resets the iterator to the beginning of the data."""
        #print('call reset()',self.shuffle,self.seq[:5])
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.seq)
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()

    def num_samples(self):
      return len(self.seq)

    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
            while True:
                if self.cur >= len(self.seq):
                    raise StopIteration
                idx = self.seq[self.cur]
                self.cur += 1
                if self.imgrec is not None:
                    s = self.imgrec.read_idx(idx)
                    if self.header0[1]:
                        header, img = recordio.unpack(s)
                    else:
                        header, img = recordio.unpack_img(s)
                    label = header.label
                    #print("sample img label ",np.shape(np.asarray(img)),header.label)
                    if not isinstance(label, numbers.Number):
                        label = label[0]
                    return label, img, None, None
                else:
                    label, fname, bbox, landmark = self.imglist[idx]
                    return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None

    def brightness_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        src *= alpha
        return src

    def contrast_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        src *= alpha
        src += gray
        return src

    def saturation_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        src *= alpha
        src += gray
        return src

    def color_aug(self, img, x):
        augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
        random.shuffle(augs)
        for aug in augs:
            #print(img.shape)
            img = aug(img, x)
            #print(img.shape)
        return img

    def mirror_aug(self, img):
        _rd = random.randint(0,1)
        if _rd==1:
            for c in range(img.shape[2]):
                img[:,:,c] = np.fliplr(img[:,:,c])
        return img

    def resize_img(self,img_numpy):
        img = img_numpy.asnumpy()
        if img.shape[0] != self.data_shape[1]  or img.shape[1] !=self.data_shape[2]:
            img = cv2.resize(img,(self.data_shape[2],self.data_shape[1]))
        else:
            pass
        return nd.array(img)


    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample()
                if self.header0[1]:
                    _data = self.imdecode(s)
                else:
                    _data = nd.array(s)
                #print("sample img shape ",np.shape(_data))
                _data = self.resize_img(_data)
                if self.rand_mirror:
                    _rd = random.randint(0,1)
                    if _rd==1:
                        _data = mx.ndarray.flip(data=_data, axis=1)
                if self.nd_mean is not None:
                    _data = _data.astype('float32')
                    _data -= self.nd_mean
                    _data *= 0.0078125
                if self.cutoff>0:
                    centerh = random.randint(0, _data.shape[0]-1)
                    centerw = random.randint(0, _data.shape[1]-1)
                    half = self.cutoff//2
                    starth = max(0, centerh-half)
                    endh = min(_data.shape[0], centerh+half)
                    startw = max(0, centerw-half)
                    endw = min(_data.shape[1], centerw+half)
                    _data = _data.astype('float32')
                    #print(starth, endh, startw, endw, _data.shape)
                    _data[starth:endh, startw:endw, :] = 127.5
                data = [_data]
                #print("next data shpe ",np.shape(_data))
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    #s_data = datum.asnumpy()
                    #s_data /= 0.0078125
                    #s_data +=127.5
                    #s_data = s_data.astype(np.uint8)
                    #s_data = s_data[::-1]
                    #cv2.imshow('img',s_data)
                    #cv2.waitKey(0)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration
        #return batch_data
        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s,flag=1,to_rgb=1) #mx.ndarray
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))



if __name__ == '__main__':
    
    path_imgrec = "./FaceAnti_train.rec"
    train_dataiter = FaceImageIter(
          batch_size           = 8,
          data_shape           = (3,224,224),
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = 1,
          mean                 = 127.5,
          cutoff               = 0,
      )
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    for i in range(2):
        print("batch num ",i)
        batch_data = train_dataiter.next()
        print("batch ",train_dataiter.provide_data)
    '''
    record = mx.recordio.MXRecordIO('test.rec', 'w')
    label = 4 # label can also be a 1-D array, for example: label = [1,2,3]
    id = 2574
    header = mx.recordio.IRHeader(0, label, id, 0)
    img = cv2.imread('test.jpg')
    packed_s = mx.recordio.pack_img(header, img)
    record.write(packed_s)
    record = mx.recordio.MXRecordIO('test.rec', 'r')
    item = record.read()
    header, img = mx.recordio.unpack_img(item)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    '''