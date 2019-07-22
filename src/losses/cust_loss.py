# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/05/29 17:09
#project: face attributes
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face attributes
####################################################
from __future__ import print_function,absolute_import
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.ndarray import NDArray
from mxnet.gluon import HybridBlock
import numpy as np
# pylint: disable=arguments-differ
"""Custom losses.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
""" 
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like


class Sigmoid(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        x = in_data[0].asnumpy()
        y = 1.0 / (1.0 + np.exp(-x))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        y = out_data[0].asnumpy()
        dy = out_grad[0].asnumpy()
        dx = dy*(1.0 - y)*y
        self.assign(in_grad[0], req[0], mx.nd.array(dx))

@mx.operator.register("sigmoid")  # register with name "sigmoid"
class SigmoidProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SigmoidProp, self).__init__(True)

    def list_arguments(self):
        #  this can be omitted if you only have 1 input.
        return ['data']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        """Calculate output shapes from input shapes. This can be
        omited if all your inputs and outputs have the same shape.

        in_shapes : list of shape. Shape is described by a tuple of int.
        """
        data_shape = in_shapes[0]
        output_shape = data_shape
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return Sigmoid()

__all__ = ['FocalLoss', 'SSDMultiBoxLoss', 'YOLOV3Loss',
           'MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'DistillationSoftmaxCrossEntropyLoss']

class FocalLoss(Loss):
    """Focal Loss for inbalanced classification.
    Focal loss was described in https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    alpha : float, default 0.25
        The alpha which controls loss curve.
    gamma : float, default 2
        The gamma which controls loss curve.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.
    num_class : int
        Number of classification categories. It is required is `sparse_label` is `True`.
    eps : float
        Eps to avoid numerical issue.
    size_average : bool, default True
        If `True`, will take mean of the output loss on every axis except `batch_axis`.

    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as label. For example, if label has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).
    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """
    def __init__(self, axis=-1, alpha=0.25, gamma=2, sparse_label=True,
                 from_logits=False, batch_axis=0, weight=None, num_class=None,
                 eps=1e-12, size_average=True, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._sparse_label = sparse_label
        if sparse_label and (not isinstance(num_class, int) or (num_class < 1)):
            raise ValueError("Number of class > 0 must be provided if sparse label is used.")
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Loss forward"""
        if not self._from_logits:
            pred = F.sigmoid(pred)
        if self._sparse_label:
            one_hot = F.one_hot(label, self._num_class)
        else:
            one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)
        t = F.ones_like(one_hot)
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        loss = -alpha * ((1 - pt) ** self._gamma) * F.log(F.minimum(pt + self._eps, 1))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            return F.mean(loss, axis=self._batch_axis, exclude=True)
        else:
            return F.sum(loss, axis=self._batch_axis, exclude=True)


class SigmoidEntropyLoss(Loss):
    r"""The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

    BCE loss is useful when training logistic regression. If `from_sigmoid`
    is False (default), this loss computes:

    .. math::

        prob = \frac{1}{1 + \exp(-{pred})}

        L = - \sum_i {label}_i * \log({prob}_i) * pos\_weight +
            (1 - {label}_i) * \log(1 - {prob}_i)

    If `from_sigmoid` is True, this loss computes:

    .. math::

        L = - \sum_i {label}_i * \log({pred}_i) * pos\_weight +
            (1 - {label}_i) * \log(1 - {pred}_i)

    A tensor `pos_weight > 1` decreases the false negative count, hence increasing
    the recall.
    Conversely setting `pos_weight < 1` decreases the false positive count and
    increases the precision.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and BCE together, which is more numerically
        stable through log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with values in range `[0, 1]`. Must have the
          same size as `pred`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).
        - **pos_weight**: a weighting tensor of positive examples. Must be a vector with length
          equal to the number of classes.For example, if pred has shape (64, 10),
          pos_weight should have shape (1, 10).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, pred, label,  pos_weight=None,sample_weight=None,):
        label = _reshape_like(F, label, pred)
        if not self._from_sigmoid:
            if pos_weight is None:
                # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
                loss = F.relu(pred) - pred * label + \
                    F.Activation(-F.abs(pred), act_type='softrelu')
            else:
                # We use the stable formula: x - x * z + (1 + z * pos_weight - z) * \
                #    (log(1 + exp(-abs(x))) + max(-x, 0))
                log_weight = 1 + F.broadcast_mul(pos_weight - 1, label)
                loss = pred - pred * label + log_weight * \
                       (F.Activation(-F.abs(pred), act_type='softrelu') + F.relu(-pred))
        else:
            eps = 1e-12
            if pos_weight is None:
                loss = -(F.log(pred + eps) * label
                         + F.log(1. - pred + eps) * (1. - label))
            else:
                loss = -(F.broadcast_mul(F.log(pred + eps) * label, pos_weight)
                         + F.log(1. - pred + eps) * (1. - label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)



if __name__=='__main__':
    x = mx.nd.array([0, 1, 2, 3])
    # attach gradient buffer to x for autograd
    x.attach_grad()
    # forward in a record() section to save computation graph for backward
    # see autograd tutorial to learn more.
    with autograd.record():
        y = mx.nd.Custom(x, op_type='sigmoid')
    print(y)
    # call backward computation
    y.backward()
    # gradient is now saved to the grad buffer we attached previously
    print(x.grad)