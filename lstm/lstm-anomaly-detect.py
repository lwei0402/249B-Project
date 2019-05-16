""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the TensorFlow backend
The basic idea is to detect anomalies in a time-series.
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import arange, sin, pi, random
import os
from keras.utils import plot_model
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D

#Using CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

#Setting random seed
np.random.seed(1234)

# Global hyper-parameters
sequence_length = 100
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 100
batch_size = 64


def dropin(X, y):
    """ The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)

def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def get_split_prep_data(train_start, train_end,
                          test_start, test_end):
    data = np.loadtxt('data/new.txt', delimiter='\n', unpack=True)


    print("Length of Data", len(data))

    for i in range(len(data)-1):
    	if data[i] > 4.25 or data[i] < -1:
    		data[i] = data[i-1]
    # plt.plot(data)
    # plt.show()
    # train data
    print ("Creating train data...")

    result = []
    for index in range(train_start, train_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print ("Mean of train data : ", result_mean)
    print ("Train data shape  : ", result.shape)

    train = result[train_start:train_end, :]
    np.random.shuffle(train)  # shuffles in-place
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_train, y_train = dropin(X_train, y_train)

    # test data
    print ("Creating test data...")

    result = []
    for index in range(test_start, test_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print ("Mean of test data : ", result_mean)
    print ("Test data shape  : ", result.shape)

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test


def build_model():
    model = Sequential()

    ###Three LSTM
    # layers = {'input': 1, 'hidden1': 64, 'hidden2': 64, 'hidden3': 64, 'output': 1}

    # model.add(LSTM(
    #         input_length=sequence_length - 1,
    #         input_dim=layers['input'],
    #         output_dim=layers['hidden1'],
    #         return_sequences=True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #         layers['hidden2'],
    #         return_sequences=True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #         layers['hidden3'],
    #         return_sequences=False))
    # model.add(Dropout(0.2))

    # model.add(Dense(
    #         output_dim=layers['output']))
    # model.add(Activation("linear"))


    ###two LSTM + two FC
    # layers = {'input': 1, 'hidden1': 128, 'hidden2': 128, 'hidden3': 128, 'hidden4': 128, 'output': 1}
    # model.add(LSTM(
    #         input_length=sequence_length - 1,
    #         input_dim=layers['input'],
    #         output_dim=layers['hidden1'],
    #         return_sequences=True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #         layers['hidden2'],
    #         return_sequences=False))
    # model.add(Dropout(0.2))

    # model.add(Dense(
    #         output_dim=layers['hidden3']))
    # model.add(Activation("relu"))

    # model.add(Dense(
    #         output_dim=layers['hidden4']))
    # model.add(Activation("relu"))

    # model.add(Dense(
    #         output_dim=layers['output']))
    # model.add(Activation("linear"))

    ###CNN Model
    # model.add(Convolution1D(input_shape = (99,1), 
    #                         nb_filter=64,
    #                         filter_length=4,
    #                         activation='relu'))
    # model.add(MaxPooling1D(pool_length=2))

    # model.add(Convolution1D(input_shape = (99,1), 
    #                         nb_filter=64,
    #                         filter_length=4,
    #                         activation='relu'))
    # model.add(MaxPooling1D(pool_length=2))

    # model.add(Dropout(0.2))
    # model.add(Flatten())

    # model.add(Dense(256))
    # model.add(Dropout(0.2))
    # model.add(Activation('relu'))

    # model.add(Dense(1))
    # model.add(Activation('linear'))

    #CNN-LSTM
    # model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(99, 1)))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    # model.add(TimeDistributed(Flatten()))
    # model.add(LSTM(64, activation='relu'))

    # model.add(Dense(1))
    # model.add(Activation('linear'))


    # model.add(Convolution1D(input_shape = (99,1), 
    #                         nb_filter=64,
    #                         filter_length=4,
    #                         activation='relu'))
    # model.add(MaxPooling1D(pool_length=2))

    model.add(Convolution1D(input_shape = (99,1), 
                            nb_filter=64,
                            filter_length=4,
                            activation='relu'))
    model.add(MaxPooling1D(pool_length=2))

    model.add(Dropout(0.2))
    model.add(TimeDistributed(Flatten()))

    # model.add(LSTM(128,return_sequences=True,activation='relu'))
    # model.add(Dropout(0.2))
    model.add(LSTM(128,return_sequences=False,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation('linear'))

    start = time.time()
    # model.compile(loss="mse", optimizer="rmsprop")
    model.compile(loss="mae", optimizer="adam")
    print ("Compilation Time : ", time.time() - start)
    plot_model(model, to_file='model.png',show_shapes=True)
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()

    if data is None:
        print ('Loading data... ')
        # train on first 700 samples and test on next 300 samples (has anomaly)
        X_train, y_train, X_test, y_test = get_split_prep_data(0, 4000, 4000, 7000)
    else:
        X_train, y_train, X_test, y_test = data

    print ('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model()

    try:
        print("Training...")
        history = model.fit(
                X_train, y_train,
                batch_size=batch_size, nb_epoch=epochs, validation_split=0.05)
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        print("Predicting...")
        predicted = model.predict(X_test)
        print("Reshaping predicted")
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print("prediction exception")
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        plt.figure(1)
        plt.subplot(311)
        plt.title("Actual Test Signal w/Anomalies")
        plt.plot(y_test[:len(y_test)], 'b')
        plt.subplot(312)
        plt.title("Predicted Signal")
        plt.plot(predicted[:len(y_test)], 'g')
        plt.subplot(313)
        plt.title("Squared Error")
        mse = ((y_test - predicted) ** 2)
        plt.plot(mse, 'r')
        plt.show()
    except Exception as e:
        print("plotting exception")
        print (str(e))
    print ('Training duration (s) : ', time.time() - global_start_time)

    return model, y_test, predicted

run_network()
