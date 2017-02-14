# Decoding: 

### A python package that includes many methods for decoding neural activity

The package contains a mixture of classic decoding methods (Wiener Filter, Wiener Cascade, Kalman Filter) and modern machine learning methods (XGBoost, Dense Neural Network, Recurrent Neural Net, GRU, LSTM).

The decoders are currently designed to predict continuously valued output. In the future, we will modify the functions to also allow classification.

This package accompanies a manuscript (soon to be released) that compares the performance of these methods on several datasets.

## Getting started
We have included jupyter notebooks that provide detailed examples of how to use the decoders. The file "" is for the Kalman filter decoder and the file "" is for all other decoders.

Here we provide a basic example:

```python
import numpy as np
```

## What's Included
There are 3 files with functions. An overview of the functions are below. More details can be found in the comments within the files.

**decoders.py:** This file provides all of the decoders. Each decoder is a class with functions "fit" and "predict".
- WienerFilterDecoder
- WienerCascadeDecoder
- KalmanFilterDecoder
- XGBoostDecoder
- DenseNNDecoder
- SimpleRNNDecoder
- GRUDecoder
- LSTMDecoder

When designing the XGBoost and neural network decoders, there were many additional parameters that could have been utilized (e.g. regularization). To simplify ease of use, we only included parameters that were sufficient for producing good fits.

**metrics.py:** The file has functions for metrics to evaluate model fit. It currently has functions to calculate:
 - $ R^2 $
 - ![equation](https://latex.codecogs.com/gif.latex?%24%5Crho%24)
 
**preprocessing_funcs.py** The file contains functions for preprocessing data that may be useful for putting the neural activity and outputs in the correct format for our decoding functions
 - get_spike_history:

## Dependencies
In order to run all the decoders based on neural networks, you need to install [Keras] (https://keras.io/#installation) <br>
In order to run the XGBoost Decoder, you need to install [XGBoost] (https://pypi.python.org/pypi/xgboost/) <br>
In order to run the Wiener Filter or Wiener Cascade, you will need [scikit-learn] (http://scikit-learn.org/stable/install.html).
