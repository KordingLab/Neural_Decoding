############### IMPORT PACKAGES ##################

import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter

#Used for naive bayes decoder
try:
    import statsmodels.api as sm
except ImportError:
    print("\nWARNING: statsmodels is not installed. You will be unable to use the Naive Bayes Decoder")
    pass
try:
    import math
except ImportError:
    print("\nWARNING: math is not installed. You will be unable to use the Naive Bayes Decoder")
    pass
try:
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    from scipy.stats import norm
    from scipy.spatial.distance import cdist
except ImportError:
    print("\nWARNING: scipy is not installed. You will be unable to use the Naive Bayes Decoder")
    pass



#Import scikit-learn (sklearn) if it is installed
try:
    from sklearn import linear_model #For Wiener Filter and Wiener Cascade
    from sklearn.svm import SVR #For support vector regression (SVR)
except ImportError:
    print("\nWARNING: scikit-learn is not installed. You will be unable to use the Wiener Filter or Wiener Cascade Decoders")
    pass

#Import XGBoost if the package is installed
try:
    import xgboost as xgb #For xgboost
except ImportError:
    print("\nWARNING: Xgboost package is not installed. You will be unable to use the xgboost decoder")
    pass

#Import functions for Keras if Keras is installed
#Note that Keras has many more built-in functions that I have not imported because I have not used them
#But if you want to modify the decoders with other functions (e.g. regularization), import them here
try:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
except ImportError:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass


##################### DECODER FUNCTIONS ##########################



##################### WIENER FILTER ##########################

class WienerFilterDecoder(object):

    """
    Class for the Wiener Filter Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn linear regression.
    """

    def __init__(self):
        return


    def fit(self,X_flat_train,y_train):

        """
        Train Wiener Filter Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        self.model=linear_model.LinearRegression() #Initialize linear regression model
        self.model.fit(X_flat_train, y_train) #Train the model


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted=self.model.predict(X_flat_test) #Make predictions
        return y_test_predicted




##################### WIENER CASCADE ##########################

class WienerCascadeDecoder(object):

    """
    Class for the Wiener Cascade Decoder

    Parameters
    ----------
    degree: integer, optional, default 3
        The degree of the polynomial used for the static nonlinearity
    """

    def __init__(self,degree=3):
         self.degree=degree


    def fit(self,X_flat_train,y_train):

        """
        Train Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs=y_train.shape[1] #Number of outputs
        models=[] #Initialize list of models (there will be a separate model for each output)
        for i in range(num_outputs): #Loop through outputs
            #Fit linear portion of model
            regr = linear_model.LinearRegression() #Call the linear portion of the model "regr"
            regr.fit(X_flat_train, y_train[:,i]) #Fit linear
            y_train_predicted_linear=regr.predict(X_flat_train) # Get outputs of linear portion of model
            #Fit nonlinear portion of model
            p=np.polyfit(y_train_predicted_linear,y_train[:,i],self.degree)
            #Add model for this output (both linear and nonlinear parts) to the list "models"
            models.append([regr,p])
        self.model=models


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs=len(self.model) #Number of outputs being predicted. Recall from the "fit" function that self.model is a list of models
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix that contains predicted outputs
        for i in range(num_outputs): #Loop through outputs
            [regr,p]=self.model[i] #Get the linear (regr) and nonlinear (p) portions of the trained model
            #Predictions on test set
            y_test_predicted_linear=regr.predict(X_flat_test) #Get predictions on the linear portion of the model
            y_test_predicted[:,i]=np.polyval(p,y_test_predicted_linear) #Run the linear predictions through the nonlinearity to get the final predictions
        return y_test_predicted



##################### KALMAN FILTER ##########################

class KalmanFilterDecoder(object):

    """
    Class for the Kalman Filter Decoder

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.

    Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
    with the exception of the addition of the parameter C.
    The original implementation has previously been coded in Matlab by Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)
    """

    def __init__(self,C=1):
        self.C=C


    def fit(self,X_kf_train,y_train):

        """
        Train Kalman Filter Decoder

        Parameters
        ----------
        X_kf_train: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted
        """

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al, 2003):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_train.T)
        Z=np.matrix(X_kf_train.T)

        #number of time bins
        nt=X.shape[1]

        #Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        #In our case, this is the transition from one kinematic state to the next
        X2 = X[:,1:]
        X1 = X[:,0:nt-1]
        A=X2*X1.T*inv(X1*X1.T) #Transition matrix
        W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        #Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        #In our case, this is the transformation from kinematics to spikes
        H = Z*X.T*(inv(X*X.T)) #Measurement matrix
        Q = ((Z - H*X)*((Z - H*X).T)) / nt #Covariance of measurement matrix
        params=[A,W,H,Q]
        self.model=params

    def predict(self,X_kf_test,y_test):

        """
        Predict outcomes using trained Kalman Filter Decoder

        Parameters
        ----------
        X_kf_test: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.

        y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The predicted outputs
        """

        #Extract parameters
        A,W,H,Q=self.model

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_test.T)
        Z=np.matrix(X_kf_test.T)

        #Initializations
        num_states=X.shape[0] #Dimensionality of the state
        states=np.empty(X.shape) #Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m=np.matrix(np.zeros([num_states,num_states]))
        P=np.matrix(np.zeros([num_states,num_states]))
        state=X[:,0] #Initial state
        states[:,0]=np.copy(np.squeeze(state))

        #Get predicted state for every time bin
        for t in range(X.shape[1]-1):
            #Do first part of state update - based on transition matrix
            P_m=A*P*A.T+W
            state_m=A*state

            #Do second part of state update - based on measurement matrix
            K=P_m*H.T*inv(H*P_m*H.T+Q) #Calculate Kalman gain
            P=(np.matrix(np.eye(num_states))-K*H)*P_m
            state=state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1]=np.squeeze(state) #Record state at the timestep
        y_test_predicted=states.T
        return y_test_predicted









##################### DENSE (FULLY-CONNECTED) NEURAL NETWORK ##########################

class DenseNNDecoder(object):

    """
    Class for the dense (fully-connected) neural network decoder

    Parameters
    ----------

    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give you a single hidden layer with 400 units)
        If you want multiple layers, input a vector (e.g. units=[400,200]) will give you 2 hidden layers with 400 and 200 units, repsectively.
        The vector can either be a list or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose

         #If "units" is an integer, put it in the form of a vector
         try: #Check if it's a vector
             units[0]
         except: #If it's not a vector, create a vector of the number of units for each layer
             units=[units]
         self.units=units

         #Determine the number of hidden layers (based on "units" that the user entered)
         self.num_layers=len(units)

    def fit(self,X_flat_train,y_train):

        """
        Train DenseNN Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add first hidden layer
        model.add(Dense(self.units[0],input_dim=X_flat_train.shape[1])) #Add dense layer
        model.add(Activation('relu')) #Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout!=0: model.add(Dropout(self.dropout))  #Dropout some units if proportion of dropout != 0

        #Add any additional hidden layers (beyond the 1st)
        for layer in range(self.num_layers-1): #Loop through additional layers
            model.add(Dense(self.units[layer+1])) #Add dense layer
            model.add(Activation('relu')) #Add nonlinear (tanh) activation
            if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units if proportion of dropout != 0

        #Add dense connections to all outputs
        model.add(Dense(y_train.shape[1])) #Add final dense layer (connected to outputs)

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy']) #Set loss function and optimizer
        model.fit(X_flat_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test) #Make predictions
        return y_test_predicted




##################### SIMPLE RECURRENT NEURAL NETWORK ##########################

class SimpleRNNDecoder(object):

    """
    Class for the simple recurrent neural network decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

        """
        Train SimpleRNN Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add recurrent layer
        model.add(SimpleRNN(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout,activation='relu')) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained SimpleRNN Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted



##################### GATED RECURRENT UNIT (GRU) DECODER ##########################

class GRUDecoder(object):

    """
    Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

        """
        Train GRU Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add recurrent layer
        model.add(GRU(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained GRU Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted



#################### LONG SHORT TERM MEMORY (LSTM) DECODER ##########################

class LSTMDecoder(object):

    """
    Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add recurrent layer
        model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted



##################### EXTREME GRADIENT BOOSTING (XGBOOST) ##########################

class XGBoostDecoder(object):

    """
    Class for the XGBoost Decoder

    Parameters
    ----------
    max_depth: integer, optional, default=3
        the maximum depth of the trees

    num_round: integer, optional, default=300
        the number of trees that are fit

    eta: float, optional, default=0.3
        the learning rate

    gpu: integer, optional, default=-1
        if the gpu version of xgboost is installed, this can be used to select which gpu to use
        for negative values (default), the gpu is not used
    """

    def __init__(self,max_depth=3,num_round=300,eta=0.3,gpu=-1):
        self.max_depth=max_depth
        self.num_round=num_round
        self.eta=eta
        self.gpu=gpu

    def fit(self,X_flat_train,y_train):

        """
        Train XGBoost Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """


        num_outputs=y_train.shape[1] #Number of outputs

        #Set parameters for XGBoost
        param = {'objective': "reg:linear", #for linear output
            'eval_metric': "logloss", #loglikelihood loss
            'max_depth': self.max_depth, #this is the only parameter we have set, it's one of the way or regularizing
            'eta': self.eta,
            'seed': 2925, #for reproducibility
            'silent': 1}
        if self.gpu<0:
            param['nthread'] = -1 #with -1 it will use all available threads
        else:
            param['gpu_id']=self.gpu
            param['updater']='grow_gpu'

        models=[] #Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            dtrain = xgb.DMatrix(X_flat_train, label=y_train[:,y_idx]) #Put in correct format for XGB
            bst = xgb.train(param, dtrain, self.num_round) #Train model
            models.append(bst) #Add fit model to list of models

        self.model=models


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained XGBoost Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        dtest = xgb.DMatrix(X_flat_test) #Put in XGB format
        num_outputs=len(self.model) #Number of outputs
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): #Loop through outputs
            bst=self.model[y_idx] #Get fit model for this output
            y_test_predicted[:,y_idx] = bst.predict(dtest) #Make prediction
        return y_test_predicted


##################### SUPPORT VECTOR REGRESSION ##########################

class SVRDecoder(object):

    """
    Class for the Support Vector Regression (SVR) Decoder
    This simply leverages the scikit-learn SVR

    Parameters
    ----------
    C: float, default=3.0
        Penalty parameter of the error term

    max_iter: integer, default=-1
        the maximum number of iteraations to run (to save time)
        max_iter=-1 means no limit
        Typically in the 1000s takes a short amount of time on a laptop
    """

    def __init__(self,max_iter=-1,C=3.0):
        self.max_iter=max_iter
        self.C=C
        return


    def fit(self,X_flat_train,y_train):

        """
        Train SVR Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs=y_train.shape[1] #Number of outputs
        models=[] #Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            model=SVR(C=self.C, max_iter=self.max_iter) #Initialize SVR model
            model.fit(X_flat_train, y_train[:,y_idx]) #Train the model
            models.append(model) #Add fit model to list of models
        self.model=models


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs=len(self.model) #Number of outputs
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): #Loop through outputs
            model=self.model[y_idx] #Get fit model for that output
            y_test_predicted[:,y_idx]=model.predict(X_flat_test) #Make predictions
        return y_test_predicted




#GLM helper function for the NaiveBayesDecoder
def glm_run(Xr, Yr, X_range):

    X2 = sm.add_constant(Xr)

    poiss_model = sm.GLM(Yr, X2, family=sm.families.Poisson())
    try:
        glm_results = poiss_model.fit()
        Y_range=glm_results.predict(sm.add_constant(X_range))
    except np.linalg.LinAlgError:
        print("\nWARNING: LinAlgError")
        Y_range=np.mean(Yr)*np.ones([X_range.shape[0],1])

    return Y_range


class NaiveBayesDecoder(object):

    """
    Class for the Naive Bayes Decoder

    Parameters
    ----------
    encoding_model: string, default='quadratic'
        what encoding model is used

    res:int, default=100
        resolution of predicted values
        This is the number of bins to divide the outputs into (going from minimum to maximum)
        larger values will make decoding slower
    """

    def __init__(self,encoding_model='quadratic',res=100):
        self.encoding_model=encoding_model
        self.res=res
        return

    def fit(self,X_b_train,y_train):

        """
        Train Naive Bayes Decoder

        Parameters
        ----------
        X_b_train: numpy 2d array of shape [n_samples,n_neurons]
            This is the neural training data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted (training data)
        """

        #### FIT TUNING CURVE ####
        #First, get the output values (x/y position or velocity) that we will be creating tuning curves over
        #Create the range for x and y (position/velocity) values
        input_x_range=np.arange(np.min(y_train[:,0]),np.max(y_train[:,0])+.01,np.round((np.max(y_train[:,0])-np.min(y_train[:,0]))/self.res))
        input_y_range=np.arange(np.min(y_train[:,1]),np.max(y_train[:,1])+.01,np.round((np.max(y_train[:,1])-np.min(y_train[:,1]))/self.res))
        #Get all combinations of x/y values
        input_mat=np.meshgrid(input_x_range,input_y_range)
        #Format so that all combinations of x/y values are in 2 columns (first column x, second column y). This is called "input_xy"
        xs=np.reshape(input_mat[0],[input_x_range.shape[0]*input_y_range.shape[0],1])
        ys=np.reshape(input_mat[1],[input_x_range.shape[0]*input_y_range.shape[0],1])
        input_xy=np.concatenate((xs,ys),axis=1)

        #If quadratic model:
        #   -make covariates have squared components and mixture of x and y
        #   -do same thing for "input_xy", which are the values for creating the tuning curves
        if self.encoding_model=='quadratic':
            input_xy_modified=np.empty([input_xy.shape[0],5])
            input_xy_modified[:,0]=input_xy[:,0]**2
            input_xy_modified[:,1]=input_xy[:,0]
            input_xy_modified[:,2]=input_xy[:,1]**2
            input_xy_modified[:,3]=input_xy[:,1]
            input_xy_modified[:,4]=input_xy[:,0]*input_xy[:,1]
            y_train_modified=np.empty([y_train.shape[0],5])
            y_train_modified[:,0]=y_train[:,0]**2
            y_train_modified[:,1]=y_train[:,0]
            y_train_modified[:,2]=y_train[:,1]**2
            y_train_modified[:,3]=y_train[:,1]
            y_train_modified[:,4]=y_train[:,0]*y_train[:,1]

        #Create tuning curves

        num_nrns=X_b_train.shape[1] #Number of neurons to fit tuning curves for
        tuning_all=np.zeros([num_nrns,input_xy.shape[0]]) #Matrix that stores tuning curves for all neurons

        #Loop through neurons and fit tuning curves
        for j in range(num_nrns): #Neuron number

            if self.encoding_model=='linear':
                tuning=glm_run(y_train,X_b_train[:,j:j+1],input_xy)
            if self.encoding_model=='quadratic':
                tuning=glm_run(y_train_modified,X_b_train[:,j:j+1],input_xy_modified)
            #Enter tuning curves into matrix
            tuning_all[j,:]=np.squeeze(tuning)

        #Save tuning curves to be used in "predict" function
        self.tuning_all=tuning_all
        self.input_xy=input_xy

        #Get information about the probability of being in one state (position/velocity) based on the previous state
        #Here we're calculating the standard deviation of the change in state (velocity/acceleration) in the training set
        n=y_train.shape[0]
        dx=np.zeros([n-1,1])
        for i in range(n-1):
            dx[i]=np.sqrt((y_train[i+1,0]-y_train[i,0])**2+(y_train[i+1,1]-y_train[i,1])**2) #Change in state across time steps
        std=np.sqrt(np.mean(dx**2)) #dx is only positive. this gets approximate stdev of distribution (if it was positive and negative)
        self.std=std #Save for use in "predict" function

        #Get probability of being in each state - we are not using this since it did not help decoding performance
        # n_x=np.empty([input_xy.shape[0]])
        # for i in range(n):
        #     loc_idx=np.argmin(cdist(y_train[0:1,:],input_xy))
        #     n_x[loc_idx]=n_x[loc_idx]+1
        # p_x=n_x/n
        # self.p_x=p_x

    def predict(self,X_b_test,y_test):

        """
        Predict outcomes using trained tuning curves

        Parameters
        ----------
        X_b_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        y_test: numpy 2d array of shape [n_samples,n_outputs]
            The actual outputs
            This parameter is necesary for the NaiveBayesDecoder  (unlike most other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        #Get values saved in "fit" function
        tuning_all=self.tuning_all
        input_xy=self.input_xy
        std=self.std

        #Get probability of going from one state to the next
        dists = squareform(pdist(input_xy, 'euclidean')) #Distance between all states in "input_xy"
        #Probability of going from one state to the next, based on the above calculated distances
        #The probability is calculated based on the distances coming from a Gaussian with standard deviation of std
        prob_dists=norm.pdf(dists,0,std)

        #Initializations
        loc_idx= np.argmin(cdist(y_test[0:1,:],input_xy)) #The index of the first location
        num_nrns=tuning_all.shape[0] #Number of neurons
        y_test_predicted=np.empty([X_b_test.shape[0],2]) #Initialize matrix of predicted outputs
        num_ts=X_b_test.shape[0] #Number of time steps we are predicting

        #Loop across time and decode
        for t in range(num_ts):
            rs=X_b_test[t,:] #Number of spikes at this time point (in the interval we've specified including bins_before and bins_after)

            probs_total=np.ones([tuning_all[0,:].shape[0]]) #Vector that stores the probabilities of being in any state based on the neural activity (does not include probabilities of going from one state to the next)
            for j in range(num_nrns): #Loop across neurons
                lam=np.copy(tuning_all[j,:]) #Expected spike counts given the tuning curve
                r=rs[j] #Actual spike count
                probs=np.exp(-lam)*lam**r/math.factorial(r) #Probability of the given neuron's spike count given tuning curve (assuming poisson distribution)
                probs_total=np.copy(probs_total*probs) #Update the probability across neurons (probabilities are multiplied across neurons due to the independence assumption)
            prob_dists_vec=np.copy(prob_dists[loc_idx,:]) #Probability of going to all states from the previous state
            probs_final=probs_total*prob_dists_vec #Get final probability (multiply probabilities based on spike count and previous state)
            # probs_final=probs_total*prob_dists_vec*self.p_x #Get final probability when including p(x), i.e. prior about being in states, which we're not using
            loc_idx=np.argmax(probs_final) #Get the index of the current state (that w/ the highest probability)
            y_test_predicted[t,:]=input_xy[loc_idx,:] #The current predicted output

        return y_test_predicted #Return predictions
