
# coding: utf-8

#Import packages
import numpy as np
from sklearn import linear_model
import xgboost as xgb

#Import everything for keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, SimpleRNN, GRU, BatchNormalization
from keras.regularizers import l2, activity_l2, l1
from keras.callbacks import EarlyStopping

#Function to get VAF (variance accounted for)
def get_vaf(y_test,y_test_pred):

    vaf_list=[]
    for i in range(y_test.shape[1]):
        y_mean=np.mean(y_test[:,i])
        vaf=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        vaf_list.append(vaf)
    return vaf_list

### Wiener filter (Linear regression)
def lin_reg_model(X_flat_train,y_train):

    regr = linear_model.LinearRegression()
    regr.fit(X_flat_train, y_train) #Train
    return regr

## Simple RNN
def SimpleRNN_model(X_train,y_train,units=400,dropout=0,num_epochs=10,verbose=0):
    model=Sequential()
    model.add(SimpleRNN(units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=dropout,dropout_U=dropout))
    if dropout!=0:
        model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1],init='uniform'))
    model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
    model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose)
    return model

### GRU (Gated Recurrent Unit)
def GRU_model(X_train,y_train,units=400,dropout=0,num_epochs=10,verbose=0):
    model=Sequential()
    model.add(GRU(units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=dropout,dropout_U=dropout))
    if dropout!=0:
        model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1],init='uniform'))
    model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
    model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose)
    return model

### LSTM (Long Short Term Memory)
def LSTM_model(X_train,y_train,units=400,dropout=0,num_epochs=10,verbose=0):
    model=Sequential()
    model.add(LSTM(units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=dropout,dropout_U=dropout))
    if dropout!=0:
        model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1],init='uniform'))
    model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
    model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose)
    return model

### XGBoost (Extreme Gradient Boosting)
def xgb_model(X_train,y_train,max_depth=3,num_round=300):

    num_outputs=y_train.shape[1]

    #Set parameters
    param = {'objective': "reg:linear", #for linear output
        'eval_metric': "logloss", #loglikelihood loss
        'max_depth': max_depth, #this is the only parameter we have set, it's one of the way or regularizing
        'seed': 2925, #for reproducibility
        'silent': 1}
    param['nthread'] = -1 #with -1 it will use all available threads

    models=[]
    for y_idx in range(num_outputs):

        dtrain = xgb.DMatrix(X_train, label=y_train[:,y_idx])
        bst = xgb.train(param, dtrain, num_round)
        models.append(bst)


    return models

def xgb_predict(models,X_test):
    dtest = xgb.DMatrix(X_test)
    num_outputs=len(models)
    y_test_pred=np.empty([X_test.shape[0],num_outputs])
    for y_idx in range(num_outputs):
        bst=models[y_idx]
        y_test_pred[:,y_idx] = bst.predict(dtest)
    return y_test_pred



    
