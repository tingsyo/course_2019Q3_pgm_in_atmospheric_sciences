#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A VGG-like Convolutional Neural Network for Single Image Super Resolution task.
- TensorFlow 2.0 based
- Read in LR and HR data
- Initialize the CNN model
- Train and test
- Output the model
"""
import os, csv, logging, argparse, glob, h5py, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, CosineSimilarity

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2019-11-14'

#-----------------------------------------------------------------------
# Utility Functions
#-----------------------------------------------------------------------
# Load input/output data for model
def loadIOTab(srcx, srcy, dropna=False):
    import pandas as pd
    # Scan for input data
    logging.info("Reading input X from: "+ srcx)
    xfiles = []
    for root, dirs, files in os.walk(srcx): 
        for fn in files: 
            if fn.endswith('.npy'): 
                 xfiles.append({'date':fn.replace('.npy',''), 'xuri':os.path.join(root, fn)})
    xfiles = pd.DataFrame(xfiles)
    print("... read input size: "+str(xfiles.shape))
    # Scan for input data
    logging.info("Reading output Y from: "+ srcy)
    yfiles = []
    for root, dirs, files in os.walk(srcy): 
        for fn in files: 
            if fn.endswith('.npy'): 
                 yfiles.append({'date':fn.replace('.npy',''), 'yuri':os.path.join(root, fn)})
    yfiles = pd.DataFrame(yfiles)
    print("... read output size: "+str(yfiles.shape))
    # Create complete IO-data
    print("Merge input/output data.")
    iotab = pd.merge(yfiles, xfiles, on='date', sort=True)
    print("... data size after merging: "+str(iotab.shape))
    # Done
    return(iotab)

def load_sprec(flist):
    ''' Load a list a Surface Precipitation files (in npy format) into one numpy array. '''
    xdata = []
    for f in flist:
        tmp = np.load(f)                            # Load numpy array
        xdata.append(np.expand_dims(tmp, axis=2))   # Append axis to the end
    x = np.array(xdata, dtype=np.float32)
    return(x)

def data_generator(iotab, batch_size):
    ''' Data generator for batched processing. '''
    nSample = len(iotab)
    # This line is just to make the generator infinite, keras needs that
    while True:
        # Initialize the sample counter
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)                     # Correct the end-pointer for the final batch
            X = load_sprec(iotab['xuri'][batch_start:limit])    # Load X
            Y = load_sprec(iotab['yuri'][batch_start:limit])    # Load Y
            yield (X,Y)                                         # Send a tuple with two numpy arrays
            batch_start += batch_size   
            batch_end += batch_size
    # End of generator

# Function to give report
def report_sisr(y_true, y_pred):
    import pandas as pd
    import sklearn.metrics as metrics
    output = []
    # Loop through samples
    n, h, w, c = y_true.shape
    for i in range(n):
        yt = y_true[i,:,:,:].flatten()
        yp = y_pred[i,:,:,:].flatten()
        # Calculate measures
        results = {}
        results['y_true_mean'] = yt.mean()
        results['y_true_var'] = yt.var()
        results['y_pred_mean'] = yp.mean()
        results['y_pred_var'] = yp.var()
        results['rmse'] = np.sqrt(metrics.mean_squared_error(yt,yp))
        if y_pred.var()<=10e-8:
            results['corr'] = 0
        else:
            results['corr'] = np.corrcoef(yt,yp)[0,1]
        output.append(results)
    # Return results
    return(pd.DataFrame(output))

# Create cross validation splits
def create_splits(iotable, prop=0.2, shuffle=False):
    idxes = np.arange(iotable.shape[0])     # Create indexes
	if shuffle:    							# Permute indexes if specified
    	idxes = np.random.permutation(idxes)
    idx_break = int(len(idxes)*(1-prop))    # Index for the split point
    idx_train = idxes[:idx_break]
    idx_test = idxes[idx_break:]
    return((idx_train, idx_test))

#-----------------------------------------------------------------------
# The model
#-----------------------------------------------------------------------
def init_model_plaindnn(input_shape):
    """
    :Return: 
      Newly initialized model for image up-scaling.
    :param 
      int input_shape: The number of variables to use as input features.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    # blovk1: CONV -> CONV
    x = BatchNormalization(axis=-1)(inputs)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv1', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv2', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv3', padding='same')(x)
    # Output block: UP_SAMPLE -> CONV
    x = UpSampling2D((2, 2), name='upsampple')(x)
    x = Conv2D(filters=1, kernel_size=(3,3), activation='relu', name='conv4', padding='same')(x)
    out = BatchNormalization(axis=-1)(x)
    # Initialize model
    model = Model(inputs = inputs, outputs = out)
    # Define compile parameters
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae', optimizer=adam, metrics=['mse','cosine_similarity'])
    return(model)
#-----------------------------------------------------------------------
# The model
#-----------------------------------------------------------------------
def init_model_plaindnn(input_shape):
    """
    :Return: 
      Newly initialized model for image up-scaling.
    :param 
      int input_shape: The number of variables to use as input features.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    # blovk1: CONV -> CONV
    x = BatchNormalization(axis=1)(inputs)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv1', padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv2', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv3', padding='same')(x)
    # Output block: UP_SAMPLE -> CONV
    x = UpSampling2D((5, 3), name='upsampple')(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv4', padding='same')(x)
    out = BatchNormalization()(x)
    # Initialize model
    model = Model(inputs = inputs, outputs = out)
    # Define compile parameters
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=['mae','cosine_similarity'], optimizer=adam, metrics=['mse','mae','cosine_similarity'])
    return(model)

#-----------------------------------------------------------------------
def main():
    #-------------------------------
    # Configure Argument Parser
    #-------------------------------
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--rawx', '-x', help='the directory containing the low-resolution data.')
    parser.add_argument('--rawy', '-y', help='the directory containing the low-resolution data.')
    parser.add_argument('--output', '-o', help='the file to store reports.')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='number of epochs.')
    parser.add_argument('--epochs', '-e', default=1, type=int, help='number of epochs.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--random_seed', '-r', default=None, type=int, help='the random seed.')
    args = parser.parse_args()
    #-------------------------------
    # Set up logging
    #-------------------------------
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    #-------------------------------
    # IO data generation
    #-------------------------------
    iotab = loadIOTab(args.rawx, args.rawy, dropna=True)
    #-------------------------------
    # Create training/testing splits
    #-------------------------------
    idx_trains, idx_tests = create_splits(iotab)
    #-------------------------------
    # Training / testing
    #-------------------------------
    # Model initialization
    model = init_model_plaindnn((nY, nX))
    # Debug info
    logging.debug(model.summary())
    logging.info("Training data samples: "+str(len(idx_trains)))
    steps_train = np.ceil(len(idx_trains)/args.batch_size)
    logging.debug("Training data steps: " + str(steps_train))
    logging.info("Testing data samples: "+ str(len(idx_tests)))
    steps_test = np.ceil(len(idx_tests)/args.batch_size)
    logging.debug("Testing data steps: " + str(steps_test))
    # Training
    hist = model.fit_generator(data_generator(iotab.iloc[idx_trains,:], args.batch_size), 
    							steps_per_epoch=steps_train, 
    							epochs=args.epochs, 
    							max_queue_size=args.batch_size64, 
    							verbose=0)
    # Prediction
    y_pred = model.predict_generator(data_generator(iotab.iloc[idx_tests,:], args.batch_size), 
    							steps=steps_test, 
    							max_queue_size=args.batch_size, 
    							verbose=0)
    # Prepare output
    report = report_sisr(y_true, y_pred)
    # Output results
    pd.DataFrame(hist).to_csv(args.output+'_hist.csv')
    report.to_csv(args.output+'_report.csv')
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
