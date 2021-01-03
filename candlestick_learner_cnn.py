# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:06:34 2020

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
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding

os.chdir('E:/projects/candlestick learner')#working directory
from candlestick_pattern_generator import *
from utilities_func import *

## parameters need to be changed for different models and training settings
os.chdir('E:/projects/candlestick learner') 
#select model
MODEL_NAME= 'OHLC-GAF-CNN' # 'OHLC-GAF-CNN': ohlc, gafcnn / 'CULR-GAF-CNN': culr, gafcnn/ 'OHLC-CNN': ohlc, cnn

INPUT_STYLE='culr'if MODEL_NAME=='CULR-GAF-CNN' else 'ohlc' # 'ohlc'/'culr'
MODEL_TYPE='cnn' if MODEL_NAME=='OHLC-CNN' else 'gafcnn'#'gafcnn'/'cnn', gafcnn convert ohlc to gaf first, cnn use ohlc directly
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


###build CNN
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = x_t.shape[1:]))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation ='relu'))
#model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.25))

model.add(Dense(9, activation = "softmax"))

# Define the optimizer
optimizer = Adam(lr=0.001,decay=1e-6)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

### tune CNN
# callbacks setting
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping=EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
my_callbacks=[learning_rate_reduction, early_stopping]

# training
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


