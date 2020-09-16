import numpy as np
import gzip
import pandas as pd
import pickle
import os
import random
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

#function to normalize data
def normalizeData(x, length=128):
    print('Normalizing: ', x.shape)
    for i in range(x.shape[0]):
        x[i,:,0] = x[i,:,0]/np.linalg.norm(x[i,:,0], 2)
    return x

#function to change amplitude to phase
def amplitudeToPhase(x, length=128):
    xComplex = x[:,0,:] + 1j*x[:,1,:]
    xAmplitude = np.abs(xComplex)
    xAngle = np.arctan2(x[:,1,:],x[:,0,:])/np.pi
    xAmplitude = np.reshape(xAmplitude, (-1,1,length))
    xAngle = np.reshape(xAngle, (-1,1,length))
    x = np.concatenate((xAmplitude, xAngle), axis = 1)
    x = np.transpose(np.array(x), (0,2,1))
    return x

#function to create datalaoder
def dataloader(fileLocation):
    with open(fileLocation, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], p.keys())))), [1,0])
    X = []
    label = []
    SNRs = []
    for mod in mods:
        for snr in snrs:
            X.append(p[(mod, snr)])
            for i in range(p[(mod,snr)].shape[0]):
                label.append(mod)
                SNRs.append(snr)
    X = np.vstack(X)
    encoder = LabelEncoder()
    encoder.fit(label)
    label = encoder.transform(label)
    label = to_categorical(label)

    X = amplitudeToPhase(X, length=128)
    X = normalizeData(X, length=128)

    #creating training, testing and validation set
    #training set contains 70% of the total data, validation and test set contains 15% each
    #SNRs are also split so as to check the classification accuracy of each SNR
    xTrain, xValidation, yTrain, yValidation, snrTrain, snrValidation = train_test_split(X, label, SNRs, test_size=0.3, shuffle=True)
    xTest, xValidation, yTest, yValidation, snrTest, snrValidation = train_test_split(xValidation, yValidation, snrValidation, test_size=0.5, shuffle=True)

    return xTrain, xValidation, xTest, yTrain, yValidation, yTest, snrTrain, snrValidation, snrTest

def classifierCreator():
    classifier = keras.models.Sequential()
    classifier.add(keras.layers.LSTM(128, batch_input_shape= (400, 128, 128), return_sequences=True, activation ='tanh'))
    classifier.add(keras.layers.LSTM(128, batch_input_shape= (400, 128, 1), activation = 'tanh' ))
    classifier.add(keras.layers.Dense(units=11, activation='softmax'))
    return classifier


classifier = classifierCreator()
xTrain, xValidation, xTest, yTrain, yValidation, yTest, snrTrain, snrValidation, snrTest = dataloader('')
classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(xTrain, yTrain, validation_data=(xValidation, yValidation), epochs=200, batch_size = 400)
classifier.save('lstmModelCategoricalLossFinal200Epochs')
