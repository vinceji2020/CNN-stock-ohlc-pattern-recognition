# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:23:31 2020

@author: Vince
"""

#import sys
#sys.path.append('E:/projects/candlestick learner')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import random
from candlestick_pattern_generator import *
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

#################### data clearning and transformation functions ################

#convert OHLC to CULR
def ohlc2culr(df):
    arr=np.array(df)
    arr1=np.zeros((10,4),float)
    for i in range(len(arr)):
        arr1[i]=[arr[i,3],arr[i,1]-max(arr[i,0],arr[i,3]),min(arr[i,0],arr[i,3])-arr[i,2],abs(arr[i,0]-arr[i,3])]
    
    return arr1

#rescale inputs to [0,1] for RNN/LSTM inputs
def rescale_ohlc(df):
    
    arr=np.array(df)
    #scale data to [0,1]
    arr=(arr-arr.min())/(arr-arr.min()).max()
    return arr

#rescale inputs to [0,1] for RNN/LSTM inputs
def rescale_culr(df):
    
    arr=np.array(df)
    #scale data to [0,1]
    i=0 # only scale close data
    arr[:,i]=(arr[:,i]-arr[:,i].min())/(arr[:,i]-arr[:,i].min()).max()
    return arr

#rescale ohlc series to [0,1] and 10*4*1 format for CNN inputs
def rescale_ohlc_cnn(df):
    
    arr=np.array(df)
    arr1 = np.zeros((10, 4, 1),float)
    #scale data to [0,1]
    arr1[:,:,0]=(arr-arr.min())/(arr-arr.min()).max()
    return arr1

#time series to gramian angular field
def ts2gaf(ts, method='summation', scale='[0,1]'):
   
    # standardize ts
    rescaled_ts = np.zeros((10, 10), float)
    min_ts, max_ts = np.min(ts), np.max(ts)
    if scale == '[0,1]':
        diff = max_ts - min_ts
        if diff != 0:
            rescaled_ts = (ts - min_ts) / diff
        
    if scale == '[-1,1]':
        diff = max_ts - min_ts
        if diff != 0:
            rescaled_ts = 2*((ts - min_ts) / diff - 0.5)
        
    # calculate Gramian Angular Matrix
    this_gam = np.zeros((10, 10), float)
    sin_ts = np.sqrt(np.clip(1 - rescaled_ts**2, 0, 1))
    if method == 'summation':
        # cos(x1+x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
        if diff !=0:
            this_gam = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
        else:
            this_gam = np.zeros((10, 10), float) - np.ones((10, 10), float)
            
    if method == 'difference':
        # sin(x1-x2) = sin(x1)cos(x2) - cos(x1)sin(x2)
        if diff !=0:
            this_gam = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)
        else:
            this_gam = np.ones((10, 10), float)
        
    return this_gam

#ohlc series to GAF
def ohlc2gaf(df, method='summation', scale='[0,1]'):
    
    arr=np.array(df)
    
    gaf = np.zeros((10, 10, 4), float)
    for i in range(4):
        gaf[:,:,i]=ts2gaf(arr[:,i], method, scale)
    
    return gaf

#prepare data to desired format before training
def pre_training_prep(x, y, model, style='ohlc', method='summation', scale='[0,1]'):
    
    test_data=x
    test_label=y
  
    test=[]
    if style=='ohlc':
        if model=='gafcnn':
            for i in range(len(test_data)):
                test.append([test_data[i,:], ohlc2gaf(test_data[i,:,:], method, scale), test_label[i,:]])       
        elif model=='cnn':    
            for i in range(len(test_data)):
                test.append([test_data[i,:], rescale_ohlc_cnn(test_data[i,:,:]), test_label[i,:]])
        elif model=='rnn':    
            for i in range(len(test_data)):
                test.append([test_data[i,:], rescale_ohlc(test_data[i,:,:]), test_label[i,:]])
    elif style=='culr':
        if model=='gafcnn':
            for i in range(len(test_data)):
                test.append([test_data[i,:], ohlc2gaf(ohlc2culr(test_data[i,:,:]), method, scale), test_label[i,:]])       
        elif model=='cnn':    
            for i in range(len(test_data)):
                test.append([test_data[i,:], rescale_ohlc_cnn(ohlc2culr(test_data[i,:,:])), test_label[i,:]])
        elif model=='rnn':    
            for i in range(len(test_data)):
                test.append([test_data[i,:], rescale_culr(ohlc2culr(test_data[i,:,:])), test_label[i,:]])
    
    #shuffle dataset
    random.shuffle(test)  

    orig_t=[]
    x_t = []
    y_t = []

    for orig, procs, target in test:  
        orig_t.append(orig)  # X is the original sequence
        x_t.append(procs)  # X is the processed sequences
        y_t.append(target)  # y is the targets/labels
        
    return np.array(orig_t), np.array(x_t), np.array(y_t)

#simulate data from pre-determined geo-brownian motion model
def pattern_gen(n, n_class):
    
    training_data=[]
    training_label=[]

    for i in range(n):
        training_data.append(np.array(morning_star()))
        training_label.append(2)
        
    for i in range(n):
        training_data.append(np.array(evening_star()))
        training_label.append(1)
    
    for i in range(n):
        training_data.append(np.array(bullish_engulfing()))
        training_label.append(4)
    
    for i in range(n):
        training_data.append(np.array(bearish_engulfing()))
        training_label.append(3)
    
    for i in range(n):
        training_data.append(np.array(trend(0.002/1e5,random.random()/1e4,10000,10)))
        training_label.append(0)
    
    training_label=to_categorical(training_label, num_classes = n_class)
    
    return np.array(training_data), np.array(training_label)



###############################################################################
######################### model diagnose functions ############################    
#plot loss and accuracy
def training_plot(history, save=False):
    
    loss=plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train","val"],loc="upper left")
    plt.show()
    if save==True:
        loss.savefig('model_loss', bbox_inches='tight', pad_inches=0)


    acc=plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","val"],loc="upper left")
    plt.show()
    if save==True:
        acc.savefig('model_acc', bbox_inches='tight', pad_inches=0)


#display confusion matrix with precision and recall
def plot_confusion_matrix(y_pred, y, ptns=range(9), save=False):
    Y_pred_classes = np.argmax(y_pred,axis = 1) # Convert predictions classes to one hot vectors 
    Y_true = np.argmax(y,axis = 1) # Convert validation observations to one hot vectors
    
    #calc confusion matrix
    cm = confusion_matrix(Y_true, Y_pred_classes) 
    cm=pd.DataFrame(cm,columns=ptns,index=ptns)
    
    #calc precision and recall
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    df=pd.DataFrame(list(zip(precision,recall)),columns=['Precision','Recall'],index=ptns)
    
    #plot confusion matrix 
    cfm=plt.figure()
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    if save==True:
        cfm.savefig('Confusion_Matrix',bbox_inches='tight', pad_inches=0)
    #plot precision and recall
    pnr=plt.figure()
    sns.heatmap(df, cmap=plt.cm.Blues, vmin=0.7, annot=True, fmt='.1%')
    plt.show()
    if save==True:
        pnr.savefig('PnR',bbox_inches='tight', pad_inches=0)
    
#display candlestick plots
def candlestick(ohlc,ttl="",save=False,name=1):
    df=pd.DataFrame(ohlc, columns=['open','high','low','close'])
    df.set_index(pd.to_datetime(df.index),inplace=True)    
    if save==True:
        mpf.plot(df,type='candle', title=ttl, savefig=f'error_obs_{name}')
    else:
        mpf.plot(df,type='candle', title=ttl)
        
# display predicted error observations
def display_errors(y_pred, y, orig, ptns=range(9), dis=10, save=False):
    Y_pred_classes = np.argmax(y_pred,axis = 1) # Convert predictions classes to one hot vectors 
    Y_true = np.argmax(y,axis = 1) # Convert validation observations to one hot vectors
    #error indexes
    errors = (Y_pred_classes - Y_true != 0)
    
    Y_pred_classes_errors = Y_pred_classes[errors]
    #Y_pred_errors = Y_pred[errors]
    Y_true_errors = Y_true[errors]
    X_errors = orig[errors]
    
    #total number of errors
    n=errors.sum()
    
    for i in range(min(n,dis)):
        ttl="Predicted: "+ str(ptns[Y_pred_classes_errors[i]])+", True: "+str(ptns[Y_true_errors[i]])
        candlestick(X_errors[i,], ttl, save, i)