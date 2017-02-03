#### IMPORT PACKAGES #####
import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter
from sklearn import linear_model #For linear regression (wiener filter)
#Import XGBoost if the package is installed
try:
    import xgboost as xgb #For xgboost
except ImportError, e:
    print("\nWARNING: Xgboost package is not installed. You will be unable to use the xgboost decoder")
    pass

#Import functions for Keras if Keras is installed
#Note that Keras has many more built-in functions that I have not imported because I have not used them
#But if you want to modify the decoders with other functions (e.g. regularization), import them here
try:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
except ImportError, e:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass



""" ###### DECODER FUNCTIONS ####### """


class WienerFilter(object):
    """Class for the Wiener Filter Decoder

    There are no parameters to set.
    """
    def __init__(self):
        return

    def fit(self,X_flat_train,y_train):
        self.model=linear_model.LinearRegression() #Initialize linear regression model
        self.model.fit(X_flat_train, y_train) #Train the model

    def predict(self,x):
        return self.model.predict(x)


class WienerCascade(object):
    """Class for the Wiener Cascade Decoder

    Parameters
    ----------
    degree: integer, optional, default 3
        The degree of the polynomial used for the nonlinearity

    """
    def __init__(self,degree=3):
         self.degree=degree

    def fit(self,X_flat_train,y_train):
        # self.fitmodel=Decoder_funcs.wiener_casc_model(x,y,self.degree)
        num_outputs=y_train.shape[1]
        models=[]
        for i in range(num_outputs):
            regr = linear_model.LinearRegression()
            regr.fit(X_flat_train, y_train[:,i]) #Fit linear
            y_train_pred_lin=regr.predict(X_flat_train)
            p=np.polyfit(y_train_pred_lin,y_train[:,i],self.degree) #Fit nonlinear
            models.append([regr,p])
        self.model=models

    def predict(self,X_flat_test):
        models=self.model
        num_outputs=len(models)
        y_test_pred_ln=np.empty([X_flat_test.shape[0],num_outputs])
        for i in range(num_outputs):
            [regr,p]=models[i]
            #Predictions on test set
            y_test_pred_lin=regr.predict(X_flat_test)
            y_test_pred_ln[:,i]=np.polyval(p,y_test_pred_lin)
        return y_test_pred_ln

##Feedforward Neural network
class DNN(object):

    """Class for the feedforward neural network decoder

    Parameters
    ----------
    num_layers: integer, optional, default 1
        Number of hidden layers in the neural network
        0 would mean that the input is directly connected to the output

    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,num_layers=1,units=400,dropout=0,num_epochs=10,verbose=0):
         self.num_layers=num_layers
         self.units=units
         self.dropout=dropout
         self.num_epochs=10
         self.verbose=verbose

    def fit(self,X_train,y_train):
        model=Sequential()
        model.add(Dense(self.units,input_dim=X_train.shape[1],init='uniform'))
        model.add(Activation('tanh'))
        if self.dropout!=0:
            model.add(Dropout(self.dropout))
        for layer in range(self.num_layers-1):
            model.add(Dense(self.units,init='uniform'))
            model.add(Activation('tanh'))
            if self.dropout!=0:
                model.add(Dropout(self.dropout))
        model.add(Dense(y_train.shape[1],init='uniform'))
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
        model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose)
        self.model=model

    def predict(self,X_test):
        return self.model.predict(X_test)
