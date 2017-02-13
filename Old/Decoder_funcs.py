
# coding: utf-8

#### IMPORT PACKAGES #####
import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter
from sklearn import linear_model #For linear regression (wiener filter)
try:
    import xgboost as xgb #For xgboost
except ImportError:
    print("\nWARNING: Xgboost package is not installed. You will be unable to use the xgboost decoder")
    pass

#Import functions for Keras
#Note that Keras has many more built-in functions that I have not imported because I have not used them
#But if you want to modify the decoders with other functions (e.g. regularization), import them here
try:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
except ImportError:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass



#### FUNCTIONS TO GET METRICS OF DECODING PERFORMANCE ####

#Function to get VAF (variance accounted for)
#Function inputs are:
#y_test - the true outputs (a matrix of size number of examples x number of outputs)
#y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)
#Function outputs are:
#vaf_list: A list of VAFs for each output
def get_vaf(y_test,y_test_pred):

    vaf_list=[] #Initialize a list that will contain the vafs for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute vaf for each output
        y_mean=np.mean(y_test[:,i])
        vaf=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        vaf_list.append(vaf) #Append VAF of this output to the list
    return vaf_list #Return the list of VAFs





#### DECODER FUNCTIONS ####



### WIENER FILTER (Linear regression) ####
#Function inputs are:
#X_flat_train - the covariates of neural data
#y_train - the outputs being predicted
#Function outputs are:
#model - the model that has been fit
def lin_reg_model(X_flat_train,y_train):

    model = linear_model.LinearRegression() #Initialize linear regression model
    model.fit(X_flat_train, y_train) #Train the model
    return model #Return the model



#### WIENER CASCADE (Linear Nonlinear model) #####
def wiener_casc_model(X_flat_train,y_train,deg=3):
    num_outputs=y_train.shape[1]
    models=[]
    for i in range(num_outputs):
        regr = linear_model.LinearRegression()
        regr.fit(X_flat_train, y_train[:,i]) #Fit linear
        y_train_pred_lin=regr.predict(X_flat_train)
        p=np.polyfit(y_train_pred_lin,y_train[:,i],deg) #Fit nonlinear
        models.append([regr,p])
    return models

def wiener_casc_predict(models,X_flat_test):
    num_outputs=len(models)
    y_test_pred_ln=np.empty([X_flat_test.shape[0],num_outputs])
    for i in range(num_outputs):
        [regr,p]=models[i]
        #Predictions on test set
        y_test_pred_lin=regr.predict(X_flat_test)
        y_test_pred_ln[:,i]=np.polyval(p,y_test_pred_lin)
    return y_test_pred_ln



##Feedforward NN - 2 hidden layers
def DNN_model(X_train,y_train,num_layers=1,units=400,dropout=0,num_epochs=10,verbose=0):
    model=Sequential()
    model.add(Dense(units,input_dim=X_train.shape[1],init='uniform'))
    model.add(Activation('tanh'))
    if dropout!=0:
        model.add(Dropout(dropout))
    for layer in range(num_layers-1):
        model.add(Dense(units,init='uniform'))
        model.add(Activation('tanh'))
        if dropout!=0:
            model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1],init='uniform'))
    model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
    model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose)
    return model




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

### Kalman Filter ###
def kf_model(zs,xs):
    #xs are the state (here, the variable we're predicting)
    #zs are the observed variable (neural data here)


    #Reformat matrices
    X=np.matrix(xs.T)
    Z=np.matrix(zs.T)

    #number of time bins (after reformatting for the lag)
    nt=X.shape[1]

    # Compute the variables required for filtering, according to Wu et al
    X2 = X[:,1:]
    X1 = X[:,0:nt-1]
    # The least-squares-optimal transformation from x_i to x_(i+1)
    A=X2*X1.T*inv(X1*X1.T)
    W=(X2-A*X1)*(X2-A*X1).T/(nt-1)

    # The least-squares-optimal transformation from x_i to z_i
    # (the transformation from position to spikes)
    H = Z*X.T*(inv(X*X.T))
    Q = ((Z - H*X)*((Z - H*X).T)) / nt
    params=[A,W,H,Q]
    return params


def kf_predict(params,zs,xs):
    #xs are the state (here, the variable we're predicting)
    #zs are the observed variable (neural data here)

    #Extract parameters
    A,W,H,Q=params

    #Reformat matrices
    X=np.matrix(xs.T)
    Z=np.matrix(zs.T)

    #Initialize
    num_states=X.shape[0]
    states=np.empty(X.shape)
    P_m=np.matrix(np.zeros([num_states,num_states]))
    P=np.matrix(np.zeros([num_states,num_states]))
    state=X[:,0]
    states[:,0]=np.copy(np.squeeze(state))

    for t in range(X.shape[1]-1):
#         P_m=np.copy(A*P*A.T+W)
#         state_m=np.copy(A*state)

        P_m=A*P*A.T+W
        state_m=A*state

        K=P_m*H.T*inv(H*P_m*H.T+Q)
        P=(np.matrix(np.eye(num_states))-K*H)*P_m
        state=state_m+K*(Z[:,t+1]-H*state_m)
#         states[:,t+1]=np.copy(np.squeeze(state))
        states[:,t+1]=np.squeeze(state)
    return states.T
