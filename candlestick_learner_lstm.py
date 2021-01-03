# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:47:09 2020

@author: Vince
"""

import pandas as pd
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding

os.chdir('E:/projects/candlestick learner')#working directory
from candlestick_pattern_generator import *
from utilities_func import *

## parameters need to be changed for different models and training settings

MODEL_NAME= 'CULR-Scaled-LSTM' # 'OHLC-Scaled-LSTM': ohlc, rnn/ 'CULR-Scaled-LSTM': culr, rnn

INPUT_STYLE='ohlc' if MODEL_NAME=='OHLC-Scaled-LSTM' else 'culr'# 'ohlc'/'culr'
MODEL_TYPE='rnn' #LSTM Model
EPOCHS = 60 
batch_size = 64
#diagnose plots output folder
DIAGNOSE_DIR=f"E:/projects/candlestick learner/model_diagnose/{MODEL_NAME}"

####preprocess data    
test_data=pd.read_pickle("label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl")#read data

#training set
orig_t, x_t, y_t=pre_training_prep(test_data['train_data'],test_data['train_label_arr'],MODEL_TYPE,INPUT_STYLE)
#validation set
orig_v, x_v, y_v=pre_training_prep(test_data['val_data'],test_data['val_label_arr'],MODEL_TYPE,INPUT_STYLE)
#test set
orig, x, y=pre_training_prep(test_data['test_data'],test_data['test_label_arr'],MODEL_TYPE,INPUT_STYLE)
#simulated test datasets
#a,b=pattern_gen(100,9)
#orig, x, y=pre_training_prep(a,b,MODEL_TYPE,INPUT_STYLE)


#### Build RNN model
model = Sequential()

model.add(LSTM(128, input_shape=(x_t.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(9, activation='softmax'))

opt = Adam(lr=0.001, decay=1e-6)

model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics=["accuracy"])

#### train RNN
# Set callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping=EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
my_callbacks=[learning_rate_reduction, early_stopping]

#training
history = model.fit(x_t, y_t, batch_size = batch_size, epochs = EPOCHS, 
                    validation_data = (x_v, y_v), verbose = 1, callbacks=my_callbacks)


############# model diagnose
#historical loss and accuracy on training and validation sets
#change directory to save plots
os.chdir(DIAGNOSE_DIR) 

training_plot(history, save=True)
#final loss and accuary on test set
score = model.evaluate(x, y, verbose=0)
print('Test loss:', score[0],'Test accuracy:', score[1])
model.save(f"{MODEL_NAME}-{int(time.time())}")

#confusion matrix, precision, recall and selected error observations
ptns=['random','evening_star','morning_star','bearish_engulfing', 'bullish_engulfing', 
      'shooting_star', 'hammer', 'harami', 'inverted_harami']
plot_confusion_matrix(model.predict(x), y, ptns, save=True)
display_errors(model.predict(x), y, orig, ptns, dis=5000,save=True)
