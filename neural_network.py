import os
import sys

# the script can be forced to use CPU
USE_GPU = False
if (USE_GPU):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from keras.layers import CuDNNLSTM as LSTM
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from keras.layers.recurrent import LSTM

from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
import numpy as np


def newModel(input_shape, num_nodes=20):
    """ Creates a new LSTM model with specified input shape and number of nodes
    
    Parameters
    ----------
        input_shape : a tuple of values based on the data. The shape is: `(batch_size, timesteps, input_dim)`
        num_nodes : number of LSTM nodes
        
    Returns
    -------
    A sequential LSTM model
    """
    
    model = Sequential()
    model.add(LSTM(num_nodes, input_shape=input_shape,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear')) # the data is not normalized, the 'linear' activation fits the case
    model.compile(loss='mae', optimizer='adam')
    return model

def train(model, x_train, y_train, lim_epochs=100, batch_size=12*8):
    """ Creates a new LSTM model with specified input shape and number of nodes
    
    Parameters
    ----------
        model : a sequential keras model to train
        x_train : A Numpy array or a list of arrays of training data with shape `(batch_size, timesteps, input_dim)`
        y_train : A list of desired outputs
        lim_epochs : the maximum number of epochs
        batch_size : an integer value to train in batches 
    """
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=lim_epochs,
        validation_split=0.1,
        shuffle=True,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20)], # use early stopping to prevent overfitting
        verbose=1
        )
    
    def show_history():
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Losses')
        plt.ylabel('Mean absolute error')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.show()
        
    show_history()
    
def evaluateModel(model, x_test, y_true):
    """Make a plot with real and predicted data
    
    Parameters
    ----------
        x_test : a numpy array of data to evaluate; the shape of the data is related to the model used
        y_true : a numpy array or a list of true values to compare 
    """
    y_pred = model.predict(x_test)[:,0] 
    plt.plot(y_true, label="true")
    plt.plot(y_pred, label="prediction")
    plt.title("True and predicted data")
    plt.xlabel("Hour")
    plt.ylabel("cnt")
    plt.legend()
    plt.show()
    print("Mean absolute error", mae(y_true, y_pred)) # np.mean(np.abs(y_pred - y_test))
    print("RMSE", np.sqrt(np.mean((y_pred-y_true)**2)))

def saveModel(model, model_fname):
    """Save the current model to file
    
    Parameters
    ----------
        model : a sequential keras model to save
        model_fname : the filename to save"""
    model_json = model.to_json()
    json_file = open(model_fname + ".json", "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights(model_fname + ".h5")
    print('"' + model_fname + '" was saved.')

def loadModel(model_fname):
    """Load an existing model
    
    Parameters
    ----------
        model_fname : a filename to save the model
        
    Returns
    -------
    A sequential LSTM model
    """
    json_file = open(model_fname + ".json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(model_fname + ".h5")
    model.compile(loss='mae', optimizer='adam')
    print('"' + model_fname + '" was loaded.')
    return model

def getPrediction(model, data):
    """Returns the predicted value of the next timestep
    
    Parameters
    ----------
        model : the model to use for the prediction
        data : a numpy array to evaluate; the shape of the data is related to the model used
        
    Returns
    -------
    A predicted Numpy float value
    """
    data = np.expand_dims(data, axis=0)
    return model.predict(data)[:,0][0]