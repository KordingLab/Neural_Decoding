# Decoding: 

### A python package that includes many methods for decoding neural activity

The package contains a mixture of classic decoding methods (Wiener Filter, Wiener Cascade, Kalman Filter) and modern machine learning methods (XGBoost, Dense Neural Network, Recurrent Neural Net, GRU, LSTM).

The decoders are currently designed to predict continuously valued output. In the future, we will modify the functions to also allow classification.

This package accompanies a manuscript (soon to be released) that compares the performance of these methods on several datasets.

## Getting started
We have included jupyter notebooks that provide detailed examples of how to use the decoders. The file "Examples_kf_decoder" is for the Kalman filter decoder and the file "Examples_all_decoders" is for all other decoders.

Here we provide a basic example where we are using a LSTM decoder. <br>
For this example we assume we have already loaded matrices:
 - "neural_data": a matrix of size "number of time bins" x "number of neurons," where each entry is the firing rate of a given neuron in a given time bin.
 - "y": the output variable that you are decoding (e.g. velocity), and is a matrix of size "number of time bins" x "number of features you are decoding."  <br>

We have provided a jupyter notebook, "Example_format_data" with an example of how to get Matlab data into this format.
<br>

First we will import the necessary functions
```python
from decoders import LSTMDecoder #Import LSTM decoder
from preprocessing_funcs import get_spikes_with_history #Import function to get the covariate matrix that includes spike history from previous bins
```
Next, we will define the time period we are using spikes from (relative to the output we are decoding)
```python
bins_before=13 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=0 #How many bins of neural data after (and including) the output are used for decoding
```

Next, we will compute the covariate matrix that includes the spike history from previous bins
```python
# Function to get the covariate matrix that includes spike history from previous bins
X=get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)
```
In this basic example, we will ignore some additional preprocessing we do in the example notebooks. Let's assume we have now divided the data into a training set (X_train, y_train) and a testing set (X_test,y_test).

We will now finally train and test the decoder:
```python
#Declare model and set parameters of the model
model_lstm=LSTMDecoder(units=400,num_epochs=5)

#Fit model
model_lstm.fit(X_train,y_train)

#Get predictions
y_test_predicted_lstm=model_lstm.predict(X_test)
```

## What's Included
There are 3 files with functions. An overview of the functions are below. More details can be found in the comments within the files.

### decoders.py:
This file provides all of the decoders. Each decoder is a class with functions "fit" and "predict".

Options for spike history/lags

Input format...X...X_flat...
Output


- **WienerFilterDecoder** 
 - The Wiener Filter is simply multiple linear regression using X_flat as an input.
 - It has no input parameters
- **WienerCascadeDecoder**
 - The Wiener Cascade (also known as a linear nonlinear model) fits a linear regression (the Wiener filter) followed by fitting a static nonlearity.
 - It has parameter *degree* (the degree of the polynomial used for the nonlinearity)
- **KalmanFilterDecoder**
 - We used a Kalman filter as implemented in [Wu et al. 2003](https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf). In the Kalman filter, the measurement was the neural spike trains, and the hidden state was the kinematics.
 - It has no input parameters
- **XGBoostDecoder**
 - We used the Extreme Gradient Boosting [XGBoost] (http://xgboost.readthedocs.io/en/latest/model.html) algorithm to relate X_flat to the outputs. XGBoost is based on the idea of boosted trees.
 - It has parameters *max_depth* (the maximum depth of the trees) and *num_round* (the number of trees that are fit)
- **DenseNNDecoder**
 - Using the Keras library, we created a dense feedforward neural network that uses X_flat to predict the outputs. It can have any number of hidden layers.
 - It has parameters *units* (the number of units in each layer), *dropout* (the proportion of units that get dropped out), *num_epochs* (the number of epochs used for training), and *verbose* (whether to display progress of the fit after each epoch)
- **SimpleRNNDecoder**
 - Using the Keras library, we created a neural network architecture where the spiking input (from matrix X) was fed into a standard recurrent neural network (RNN). The units from this recurrent layer were fully connected to the output layer. 
 - It has parameters *units*, *dropout*, *num_epochs*, and *verbose*
- **GRUDecoder**
 - All methods were the same as for the SimpleRNNDecoder, except  Gated Recurrent Units (GRUs; a more sophisticated RNN) were used rather than a traditional RNN. 
 - It has parameters *units*, *dropout*, *num_epochs*, and *verbose*
- **LSTMDecoder**
 - All methods were the same as for the SimpleRNNDecoder, except  Long Short Term Memory networks (LSTMs; a more sophisticated RNN) were used rather than a traditional RNN. 
 - It has parameters *units*, *dropout*, *num_epochs*, and *verbose*

When designing the XGBoost and neural network decoders, there were many additional parameters that could have been utilized (e.g. regularization). To simplify ease of use, we only included parameters that were sufficient for producing good fits.

### metrics.py:
The file has functions for metrics to evaluate model fit. It currently has functions to calculate:
 - ![equation](https://latex.codecogs.com/gif.latex?%24R%5E2%3D1-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%7D%5Cfrac%7B%5Cleft%28y_i-%5Cwidehat%7By_i%7D%20%5Cright%20%29%5E2%7D%7B%5Cleft%28y_i-%5Cbar%7By_i%7D%20%5Cright%20%29%5E2%7D)
 - ![equation](https://latex.codecogs.com/gif.latex?%24%5Crho%24) : The pearson correlation coefficient
 
### preprocessing_funcs.py
The file contains functions for preprocessing data that may be useful for putting the neural activity and outputs in the correct format for our decoding functions
 - **bin_spikes**: converts spike times to the number of spikes within time bins
 - **bin_output**: converts a continuous stream of outputs to the average output within time bins
 - **get_spikes_with_history**: using binned spikes as input, this function creates a covariate matrix of neural data that incorporates spike history
 
## Dependencies
In order to run all the decoders based on neural networks, you need to install [Keras] (https://keras.io/#installation) <br>
In order to run the XGBoost Decoder, you need to install [XGBoost] (https://pypi.python.org/pypi/xgboost/) <br>
In order to run the Wiener Filter or Wiener Cascade, you will need [scikit-learn] (http://scikit-learn.org/stable/install.html).
