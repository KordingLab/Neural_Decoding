# Decoding: 

## A python package that includes many methods for decoding neural activity

The package contains a mixture of classic decoding methods (Wiener Filter, Wiener Cascade, Kalman Filter) and modern machine learning methods (XGBoost, Dense Neural Network, Recurrent Neural Net, GRU, LSTM).

This package accompanies a manuscript (soon to be released) that compares the performance of these methods on several datasets.

## Getting started
We have included jupyter notebooks that provide detailed examples of how to use the decoders. The file "" is for the Kalman filter decoder and the file "" is for all other decoders.

Here we provide a basic example:



## What's Included
There are 3 files with functions

**decoders.py:**
- rnn

Architectures

metrics.py:

preprocessing_funcs.py


## Dependencies
In order to run all the decoders based on neural networks, you need to install [Keras] (https://keras.io/#installation) <br>
In order to run the XGBoost Decoder, you need to install [XGBoost] (https://pypi.python.org/pypi/xgboost/) <br>
In order to run the Wiener Filter or Wiener Cascade, you will need [scikit-learn] (http://scikit-learn.org/stable/install.html).

## Future additions
-Classification
