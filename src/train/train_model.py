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
from __future__ import print_function  # only relevant for Python 2
import os
import sys
import time
import numpy as np
import mxnet as mx
import gluoncv as gcv
import logging
from mxnet import nd, gluon, autograd
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from  cust_loss import SigmoidEntropyLoss
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


def save_gloun2model(self, prefix):
        self.net.hybridize()
        x = mx.sym.var('data')
        y = self.net(x)
        symnet = mx.symbol.load_json(y.tojson())
        params = self.net.collect_params()
        args = {}
        auxs = {}    
        for param in params.values():
            v = param._reduce()
            k = param.name
            if 'running' in k:
                auxs[k] = v
            else:
                args[k] = v            
        mod = mx.mod.Module(symbol=symnet, context=self.ctx)
        mod.bind(for_training=False, 
                 data_shapes=[('data', (1, 3, self.data_shape, self.data_shape))])
        mod.set_params(arg_params=args, aux_params=auxs)        
        mod.save_checkpoint(prefix, epoch)

def get_layer_output(symbol, arg_params, aux_params, layer_name):
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    #net = mx.symbol.Flatten(data=net)
    new_args = dict({k:arg_params[k] for k in arg_params if k in net.list_arguments()})
    new_aux = dict({k:aux_params[k] for k in aux_params if k in net.list_arguments()})
    return (net, new_args, new_aux)

def load_parms(new_sym,new_arg_params,new_aux_params,ctx):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre_trained = gluon.nn.SymbolBlock(outputs=new_sym, inputs=mx.sym.var('data'))
    net_params = pre_trained.collect_params()
    #print(net_params)
    for param in new_arg_params:
        if param in net_params:
            net_params[param]._load_init(new_arg_params[param], ctx=ctx)
    for param in new_aux_params:
        if param in net_params:
            net_params[param]._load_init(new_aux_params[param], ctx=ctx)
    return pre_trained

def get_pretrained_layer(net_train,layer_name):
    inputs = mx.sym.var('data')
    out = net_train(inputs)
    internals = out.get_internals()
    #print(internals.list_outputs())
    outputs = internals[layer_name+"_output"]
    # Create SymbolBlock that shares parameters with alexnet
    #mx.viz.plot_network(outputs, shape={'data': (1, 3, 224, 224)}).view()
    feat_model = gluon.SymbolBlock(outputs, inputs, params=net_train.collect_params())
    #x = mx.nd.random.normal(shape=(16, 3, 224, 224))
    #print(feat_model(x))
    return feat_model

def add_layer(ctx,name='sigmoid'):
    dense_layer = gluon.nn.Dense(cfgs.CLS_NUM,activation=name)
    dense_layer.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    return dense_layer

def graph(pre_trained,dense_layer):
    net = gluon.nn.HybridSequential()
    #net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(pre_trained)
        net.add(gluon.nn.Flatten())
        net.add(dense_layer)
    return net

def load_model(load_epoch):
    model_prefix = cfgs.MODEL_PREFIX
    assert model_prefix is not None
    model_prefix = os.path.join('../../models',cfgs.DATASET_NAME,model_prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, load_epoch)
    #print('Loaded model %s_%04d.params', model_prefix, load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, load_epoch)
    return (sym, arg_params, aux_params)   

def evaluate_accuracy_gluon(data_iterator,net,ctx):
    num_instance = 0
    threshold = [0.5]
    sum_metric = nd.zeros(cfgs.CLS_NUM,ctx=ctx, dtype=np.float32)
    for data, label in data_iterator:
        data = data.astype(np.float32).as_in_context(ctx)
        label = label.astype(np.float32).as_in_context(ctx)
        output = net(data)
        #prediction = nd.argmax(output, axis=1).astype(np.int32)
        prediction = nd.broadcast_greater(output,nd.array(threshold,ctx=ctx))
        num_instance += len(prediction)
        #print(predition.shape,label.shape)
        p_mask = nd.broadcast_equal(prediction,label)
        p_mask = p_mask.sum(axis=0)
        sum_metric += p_mask
    accuracy = sum_metric/num_instance
    return accuracy

def train_acc_metric(pred_data,labels,ctx):
    threshold = [0.5]
    pred_data = pred_data.astype(np.float32).as_in_context(ctx)
    labels = labels.astype(np.float32).as_in_context(ctx)
    #prediction = nd.argmax(output, axis=1).astype(np.int32)
    prediction = nd.broadcast_greater(pred_data,nd.array(threshold,ctx=ctx))
    num_instance = len(prediction)
    prediction = prediction.astype(np.float32)
    sum_metric = (prediction==labels)
    sum_metric = sum_metric.sum(axis=0)
    accuracy = sum_metric.astype(np.float32)/num_instance
    return accuracy

def learning_rate_schedule(batch_size):
    steps_epochs = cfgs.DECAY_STEP
    train_num = cfgs.DATA_NUM
    # assuming we keep partial batches, see `last_batch` parameter of DataLoader
    iterations_per_epoch = np.ceil(train_num / batch_size)
    # iterations just before starts of epochs (iterations are 1-indexed)
    steps_iterations = [s*iterations_per_epoch for s in steps_epochs]
    #print("Learning rate drops after iterations: {}".format(steps_iterations))
    logging.info("Learning rate drops after iterations: {}".format(steps_iterations))
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=0.1)
    return schedule

def get_rmse_log(net, X_train, y_train):
    """Gets root mse between the logarithms of the prediction and the truth."""
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)


def fit(net,train_loader,val_loader,**kargs):
    '''
    '''
    #ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
    num_epochs = kargs.get('epoch',2000)
    ctx = kargs.get('ctx',mx.cpu(0))
    save_epoch = kargs.get('save_epoch',10)
    load_epoch = kargs.get('load_epoch',None)
    batch_size = kargs.get('batch_size',1)
    lr = kargs.get('learning_rate',0.04)
    model_path = kargs.get('model_dir',None)
    if load_epoch is None:
        net.initialize(mx.init.Xavier(), ctx=ctx)
    schedule = learning_rate_schedule(batch_size)
    opt_optimizer = mx.optimizer.SGD(learning_rate=lr, lr_scheduler=schedule)
    #ada_optimizer = mx.optimizer.Adam(learning_rate=lr)
    trainer = gluon.Trainer(
        params=net.collect_params(),
        optimizer=opt_optimizer)
        #optimizer='sgd',
        #optimizer_params={'learning_rate': lr})
    #metric = mx.metric.Accuracy()
    #loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    loss_function = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True,batch_axis=1)
    #loss_function = gcv.loss.FocalLoss(spares_label=False,from_logits=True,batch_axis=1)
    loss_function = SigmoidEntropyLoss(from_sigmoid=True,batch_axis=1)
    global_step = 0
    total_loss = 0
    pos_factor = nd.array([2,8,8,5,8,10,10,8,3,5,2,2,8,10,5,2,2,3,5,8,5],dtype=np.float32,ctx=ctx)
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # Possibly copy inputs and labels to the GPU
            start = time.time()
            inputs = inputs.as_in_context(ctx)
            labels = labels.as_in_context(ctx)
            # The forward pass and the loss computation need to be wrapped
            # in a `record()` scope to make sure the computational graph is
            # recorded in order to automatically compute the gradients
            # during the backward pass.
            with autograd.record():
                outputs = net(inputs)
                loss_result = loss_function(outputs, labels,pos_weight=pos_factor)
            # Compute gradients by backpropagation and update the evaluation
            loss_result.backward()
            # Update the parameters by stepping the trainer; the batch size
            # is required to normalize the gradients by `1 / batch_size`.
            trainer.step(batch_size=inputs.shape[0])
            total_loss = nd.sum(loss_result)
            end = time.time()
            global_step+=1
            if global_step % cfgs.SHOW_TRAIN_INFO_INTE ==0:
                #metric.update(labels, outputs)
                #name, acc = metric.get()
                acc = train_acc_metric(outputs,labels,mx.cpu(0))
                #print('Step {}: {} = {}'.format(global_step, name, acc))
                logging.info('epoch:{} \t Batch_cost:{:3f}S \t Step:{} loss:{} \t lr:{}'.format(epoch,(end-start),global_step,total_loss.asscalar()/21.0,opt_optimizer.lr))
                #logging.info('acc: ',acc)
                print('\t \t \t rain acc--: ',acc.asnumpy())
                print('\t \t \t every loss: ',loss_result.asnumpy())
                #print("loss,",total_loss.asscalar())
            if global_step % cfgs.SMRY_ITER == 0:
                eval_acc = evaluate_accuracy_gluon(val_loader,net,ctx)
                #print('Step {}: val_acc = {}'.format(global_step, eval_acc))
                print('*** *****Test: Step: %d ' %(global_step),'\t', 'acc: ',eval_acc.asnumpy())
            
        #name, acc = metric.get()
        #logging.info("acc: ",acc)
        #logging.info("loss: ",total_loss.asscalar()/cfgs.DATA_NUM)
        #print('acc: ',acc)
        #metric.reset()
        if epoch % save_epoch ==0:
            save_path = os.path.join(model_path,cfgs.DATASET_NAME)
            if os.path.exists(save_path):
                pass
            else:
                os.mkdirs(save_path)
            save_path = os.path.join(save_path,cfgs.MODEL_PREFIX)
            net.export(save_path, epoch=epoch)
            print("************************save weights*****************",epoch)

if __name__=='__main__':
    p = nd.array([0.9,0.1,0.4,0.6,0.7,0.9,0.1,0.4,0.6,0.7,0.9,0.1,0.4,0.6,0.7,0.9,0.1,0.4,0.6,0.7,0.1],ctx=mx.gpu(0))
    l = nd.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    net=0
    a= train_acc_metric(p,l,mx.gpu(0))
    print(a)