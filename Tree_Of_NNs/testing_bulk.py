import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import time
import os
import sys
import random

dataframe = pd.read_csv("NN_data.csv",delimiter="," ,header=None)
data = dataframe.values

## Training data for leaf networks
x_trn = data[:4000,0:5]
y_trn = data[:4000,5]
x_tst = data[4001:5000,0:5]
y_tst = data[4001:5000,5]

## One hot encoding
dummy_y_trn = np_utils.to_categorical(y_trn)
dummy_y_trn = np.delete(dummy_y_trn, obj=0, axis=1)


## Function that creates our neural network
def entire_NN():
    # Creating model
    model = Sequential()

    ## Adding input layer with rectified linear activation function for input layer
    model.add(Dense(5, input_dim=5, activation='relu'))
    model.add(Dense(14,activation='relu'))

    ## Adding output layer with softmax activation function
    model.add(Dense(26, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

## Function for predicting labels using keras predict function
def predict(model,x):
    preds = np.argmax(model.predict(x), axis=-1)
    #preds = model.predict_classes(x)
    for i in range(len(preds)):
        preds[i] = preds[i] + 1
    return preds

## Function for testing accuracy of predictions
def testAccuracy(preds, labels):
    count = 0
    for i in range(len(preds)):
        if(preds[i] == labels[i]):
            count += 1
    final = 100*(count/len(labels))
    #return print("Accuracy: %.2f" % (final))
    return final

while True:
    epochs_num = random.randint(90,300)
    batch_num = random.randint(3,100)

    model = entire_NN()

    model.fit(x_trn, dummy_y_trn, epochs=epochs_num, batch_size=batch_num, verbose=0)

    predictions = predict(model,x_tst)
    x = testAccuracy(predictions,y_tst)

    bulk_epoch = open("bulk_epochs.tsv", "a")
    l_bulk_filesize = os.path.getsize("bulk_epochs.tsv")
    if(l_bulk_filesize > 0):
        bulk_epoch.write("\n")
    bulk_epoch.write("%.0f" % epochs_num)
    bulk_epoch.close()

    bulk_batch = open("bulk_batch.tsv", "a")
    b_bulk_filesize = os.path.getsize("bulk_batch.tsv")
    if(b_bulk_filesize > 0):
        bulk_batch.write("\n")
    bulk_batch.write("%.0f" % batch_num)
    bulk_batch.close()

    bulk_accuracy = open("bulk_accuracies.tsv", "a")
    b_a_filesize = os.path.getsize("bulk_accuracies.tsv")
    if(b_a_filesize > 0):
        bulk_accuracy.write("\n")
    bulk_accuracy.write("%.2f" % x)
    bulk_accuracy.close()

    time.sleep(1)
