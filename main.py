import os
import numpy as np
import argparse
import keras
import time
from confusionmatrix import ConfusionMatrix
import random
import sys

from networks import DenseNet_Transformer
from sklearn.metrics import roc_auc_score
from keras.utils import np_utils
# from keras.model import load_model
from keras.models import Model, load_model, Sequential
from keras_utils2 import TransformerBlock,DenseNet,AddPositionEmbedding
from tensorflow.keras.layers import Dense,Conv1D,GlobalAveragePooling1D,Activation,Dropout
from keras import regularizers
from keras.layers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def iterate_minibatches(inputs, targets, masks,  batchsize, shuffle=False, sort_len=True):
    """ Generate minibatches of a specific size
    Arguments:
        inputs -- numpy array of the encoded protein data. Shape: (n_samples, seq_len, n_features)
        targets -- numpy array of the targets. Shape: (n_samples,)
        masks -- numpy array of the protein masks. Shape: (n_samples, seq_len)
        batchsize -- integer, number of samples in each minibatch.
        shuffle -- boolean, shuffle the samples in the minibatches. (default=False)
        sort_len -- boolean, sort the minibatches by sequence length (faster computation, just for training). (default=True)
    Outputs:
    list of minibatches for protein sequences, targets and masks.
    """
    all_batch = []
    assert len(inputs) == len(targets)   

    len_seq = np.apply_along_axis(np.bincount, 1, masks.astype(np.int32))[:, -1]
    # print(len_seq)

    # Sort the sequences by length
    if sort_len:
        indices = np.argsort(len_seq)
    else:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)  

    # Generate minibatches list
    f_idx = len(inputs) % batchsize  
    idx_list = list(range(0, len(inputs) - batchsize + 1, batchsize))  
    last_idx = None
    if f_idx != 0:
        last_idx = idx_list[-1] + batchsize
        idx_list.append(last_idx)

    # Shuffle the minibatches
    if shuffle:
        random.shuffle(idx_list)

    # Split the data in minibatches
    for start_idx in idx_list:
        if start_idx == last_idx:
            rand_samp = batchsize - f_idx
            B = np.random.randint(len(inputs), size=rand_samp)
            excerpt = np.concatenate((indices[start_idx:start_idx + batchsize], B))
        else:
            excerpt = indices[start_idx:start_idx + batchsize]
        max_prot = np.amax(len_seq[excerpt])

        # Crop batch to maximum sequence length
        if sort_len:
            in_seq = inputs[excerpt][:, :max_prot]
            in_mask = masks[excerpt][:, :max_prot]
        else:
            in_seq = inputs[excerpt]
            in_mask = masks[excerpt]

        in_target = targets[excerpt]
        shuf_ind = np.arange(batchsize)

        # Shuffle samples within each minitbatch
        if shuffle:
            np.random.shuffle(shuf_ind)

        # Return a minibatch of each array
        yield in_seq[shuf_ind], in_target[shuf_ind], in_mask[shuf_ind]

def testOneBalance(ID):
    parser = argparse.ArgumentParser()  
    parser.add_argument('-i', '--trainset', help="npz file with traning profiles data",                       
                        default='./data/demo_train.npz')
    parser.add_argument('-t', '--testset', help="npz file with test profiles data to calculate final accuracy",
                        default='./data/demo_test.npz')
    parser.add_argument('-bs', '--batch_size',  help="Minibatch size, default = 64", default=64)
    parser.add_argument('-e', '--epochs',  help="Number of training epochs, default = 200", default=800)
    parser.add_argument('-n', '--n_filters',  help="Number of filters, default = 10", default=2)
    parser.add_argument('-lr', '--learning_rate',  help="Learning rate, default = 0.0003", default=0.0001)
    parser.add_argument('-id', '--in_dropout',  help="Input dropout, default = 0.2", default=0.2)
    parser.add_argument('-hd', '--hid_dropout',  help="Hidden layers dropout, default = 0.5", default=0.7)
    parser.add_argument('-hn', '--n_hid',  help="Number of hidden units, default = 256", default=256)
    parser.add_argument('-se', '--seed',  help="Seed for random number init., default = 123456", default=123456)
    args = parser.parse_args()

    if args.trainset == None or args.testset == None:
        parser.print_help()
        sys.stderr.write("Please specify training and test data file!\n")
        sys.exit(1)

 
    n_class = 2  
    batch_size = int(args.batch_size)
    seq_len = 1000  
    n_hid = int(args.n_hid)  
    lr = float(args.learning_rate)
    num_epochs = int(args.epochs)
    drop_per = float(args.in_dropout)
    drop_hid = float(args.hid_dropout)
    n_filt = int(args.n_filters)


    # Load data
    print("Loading data...\n")
    test_data = np.load(args.testset)
    train_data = np.load(args.trainset)

    # Test set
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    mask_test = test_data['mask_test']
    partition_test = test_data['partition']
    y_test_n = np.zeros(np.shape(y_test),dtype=int)

    # Initialize output vectors from test set
    complete_alpha = np.zeros((X_test.shape[0] ,seq_len))
    complete_context = np.zeros((X_test.shape[0] ,n_hid *2))
    complete_test = np.zeros((X_test.shape[0] ,n_class))

    # Training set
    X_train = train_data['X_train']
    y_train = train_data['y_train']

    mask_train = train_data['mask_train']
    partition = train_data['partition']

    # Number of features
    n_feat = np.shape(X_test)[2]   # n_feat=20
    X_train = X_train.reshape(X_train.shape[0], seq_len, n_feat)
    X_test = X_test.reshape(X_test.shape[0], seq_len, n_feat)    
    input_shape = (seq_len, n_feat)  # seq_len=1000

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    n_class = 2
    all_confusion_valid = []
    for i in range(1,2):
        all_confusion_valid.append(ConfusionMatrix(n_class))

    for i in range(1,2):
        print("Compilation model {}\n".format(i))
        train_index = np.where(partition != i)
        val_index = np.where(partition == i)
        X_tr = X_train[train_index].astype(np.float32)
        X_val = X_train[val_index].astype(np.float32)
        y_tr = y_train[train_index].astype(np.int32)
        y_val = y_train[val_index].astype(np.int32)
        mask_tr = mask_train[train_index].astype(np.float32)
        mask_val = mask_train[val_index].astype(np.float32)

        eps = []
        best_val_acc = 0
        best_epoch = 0
        best_predict_val = []
        best_y_test = []
        print(input_shape)
        model = DenseNet_Transformer(n_class, input_shape)
        print(model.summary())
        print(model.metrics_names)
        print("Start training\n")	
        all_train_loss =[]
        all_val_loss =[]

        batch_pre_val = []
        batch_targets = []

        noIncreaseStep = 0
        stopStep = 700
        best_test_acc = 0     
        
        for epoch in range(num_epochs):
            # Calculate epoch time
            start_time = time.time()
            if(noIncreaseStep>stopStep):
                break
            # Full pass training set
            train_err = 0
            train_acc = 0
            train_batches = 0
            confusion_train = ConfusionMatrix(n_class)
            
            for batch in iterate_minibatches(X_tr, y_tr, mask_tr, batch_size, shuffle=True, sort_len=False):
                inputs, targets, in_masks = batch               
                targets1 = np_utils.to_categorical(targets, n_class)
                tr_err, acc = model.train_on_batch(inputs, targets1)
                train_err += tr_err
                train_acc += acc
                train_batches += 1
                predict = model.predict(inputs)
                preds = np.argmax(predict, axis=-1)
                confusion_train.batch_add(targets, preds)
            train_loss = train_err / train_batches        
            train_accuracy = train_acc / train_batches
            train_accuracy1 = confusion_train.accuracy()
            train_sensitivity = confusion_train.sensitivity()
            train_specificity = confusion_train.specificity()
            cf_train = confusion_train.ret_mat()
            print(confusion_train.mat)
            print(str(epoch) + " " + str(train_loss) + " " + str(train_accuracy)+" "+ str(train_accuracy1)+" "+str(train_sensitivity)+" "+str(train_specificity))

            # Full pass validation set
            val_err = 0
            val_batches = 0
            val_acc = 0
            confusion_valid = ConfusionMatrix(n_class)

            batch_pre_val = []
            batch_targets = []
            # Generate minibatches and train on each one of them
            for batch in iterate_minibatches(X_val, y_val, mask_val, batch_size, shuffle=False, sort_len=False):             
                inputs, targets, in_masks = batch
                targets1 = np_utils.to_categorical(targets, n_class)
                err, acc = model.evaluate(inputs, targets1, verbose=0)
                val_acc += acc
                predict_val = model.predict(inputs)
                batch_pre_val.append(predict_val)
                batch_targets.append(targets)
                val_err += err
                val_batches += 1
                preds = np.argmax(predict_val, axis=-1)
                confusion_valid.batch_add(targets, preds)

            batch_pre_val = np.array(batch_pre_val).reshape((-1, 2))
            batch_targets = np.array(batch_targets).reshape((-1))

            val_loss = val_err / val_batches
            val_accuracy = val_acc / val_batches
            val_accuracy1 = confusion_valid.accuracy()
            val_sensitivity = confusion_valid.sensitivity()
            val_specificity = confusion_valid.specificity()            
            val_precision = confusion_valid.precision()
            cf_val = confusion_valid.ret_mat()

            print(val_loss, end="\t")
            print(val_accuracy, end="\t")
            print(val_accuracy1, end="\t")
            print(val_sensitivity, end="\t")
            print(val_specificity, end="\t")
            print(val_precision)

            f_val_acc = val_accuracy
            noIncreaseStep = noIncreaseStep + 1
            
            confusion_test = ConfusionMatrix(n_class)
            if f_val_acc >= best_val_acc:
                print("Good job, i like it")
                model.save('DenSec-model.h5')  # 存放最佳
                best_val_acc = val_accuracy                
                best_predict_val = predict_val                
                noIncreaseStep = 0

            batch_pre_test = []
            batch_targets = []
            y_test1 = np_utils.to_categorical(y_test, n_class)
            print(np.shape(X_test))
            err, acc = model.evaluate(X_test, y_test1, verbose=0)
            predict_test = model.predict(X_test)
            preds = np.argmax(predict_test, axis=-1)
            batch_pre_test.append(predict_test)
            batch_targets.append(y_test)
            confusion_test.batch_add(y_test, preds)                

            test_loss = err
            test_accuracy = acc
            test_accuracy1 = confusion_test.accuracy()
            test_sensitivity = confusion_test.sensitivity()
            test_specificity = confusion_test.specificity()               
            test_precision = confusion_test.precision()
            # test_MCC=confusion_test.matthews_correlation()
            # test_AUC=roc_auc_score(y_test,preds)
            cf_val = confusion_test.ret_mat()
            print(cf_val)
            print(best_test_acc,test_accuracy,epoch,best_epoch)                
            print(test_loss, end="\t")
            print(test_accuracy, end="\t")
            print(test_accuracy1, end="\t")
            print(test_sensitivity, end="\t")
            print(test_specificity, end="\t")
            # print(test_MCC,end="\t")
            # print(test_AUC,end="\t")
            print(test_precision)
                                              
            if(best_test_acc<test_accuracy):                    
                best_epoch = epoch
                best_test_acc = test_accuracy                   
                best_predict_test = predict_test
                best_y_test = y_test                

        print(best_epoch)
        print(best_val_acc)
        print(best_test_acc)
        print("ok")

import time
for ii in range(0,1):
    start = time.time()
    testOneBalance(ii)
    end = time.time()
    print("循环运行时间:%.2f秒"%(end-start))