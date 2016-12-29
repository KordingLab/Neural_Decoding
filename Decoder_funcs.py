
# coding: utf-8

#Import packages
import numpy as np
from sklearn import linear_model
# import xgboost as xgb

#Import everything for keras
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, SimpleRNN, GRU, BatchNormalization
# from keras.regularizers import l2, activity_l2, l1
# from keras.callbacks import EarlyStopping

#Function to get VAF (variance accounted for)
def get_vaf(y_test,y_test_pred):

    vaf_list=[]
    for i in range(y_test.shape[1]):
        y_mean=np.mean(y_test[:,i])
        vaf=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        vaf_list.append(vaf)
    return vaf_list

### Wiener filter (Linear regression)
def lin_reg_model(X_flat_train,y_train,X_flat_test):

    regr = linear_model.LinearRegression()
    regr.fit(X_flat_train, y_train) #Train
    y_test_pred=regr.predict(X_flat_test)
    return y_test_pred
#
# ### Simple RNN
# def SimpleRNN_model(X_train,y_train,X_test,units=400,dropout=0,num_epochs=10,verbose=0):
#     model=Sequential()
#     model.add(SimpleRNN(units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=dropout,dropout_U=dropout))
#     if dropout!=0:
#         model.add(Dropout(dropout))
#     model.add(Dense(y_test.shape[1],init='uniform'))
#     model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
#     model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose)
#     y_test_pred=model.predict(X_test)
#     return y_test_pred
#
# ### GRU (Gated Recurrent Unit)
# def GRU_model(X_train,y_train,X_test,units=400,dropout=0,num_epochs=10,verbose=0):
#     model=Sequential()
#     model.add(GRU(units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=dropout,dropout_U=dropout))
#     if dropout!=0:
#         model.add(Dropout(dropout))
#     model.add(Dense(y_test.shape[1],init='uniform'))
#     model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
#     model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose)
#     y_test_pred=model.predict(X_test)
#     return y_test_pred
#
# ### LSTM (Long Short Term Memory)
# def LSTM_model(X_train,y_train,X_test,units=400,dropout=0,num_epochs=10,verbose=0):
#     model=Sequential()
#     model.add(LSTM(units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=dropout,dropout_U=dropout))
#     if dropout!=0:
#         model.add(Dropout(dropout))
#     model.add(Dense(y_test.shape[1],init='uniform'))
#     model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
#     model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose)
#     y_test_pred=model.predict(X_test)
#     return y_test_pred

### XGBoost (Extreme Gradient Boosting)
