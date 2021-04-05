import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import time
import os
import sys
import random

dataframe = pd.read_csv("Vivek_Ranged_Data.csv",delimiter="," ,header=None)
data = dataframe.values

## Training data for leaf networks
x_trn = data[:9000,0:5]
y_trn = data[:9000,5]
x_tst = data[9001:10000,0:5]
y_tst = data[9001:10000,5]

## One hot encoding
dummy_y_trn = np_utils.to_categorical(y_trn)
dummy_y_trn = np.delete(dummy_y_trn, obj=0, axis=1)


## Function that creates our neural network
def bulk_NN(nodes, num_layers, num_input_nodes):
    # Creating model
    model = Sequential()

    model.add(Dense(num_input_nodes, input_dim=5, activation='sigmoid'))

    for i in range(num_layers):
        model.add(Dense(nodes[i], activation='sigmoid'))

    ## Adding output layer with softmax activation function
    model.add(Dense(27, activation='softmax'))
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
    num_layers = random.randint(1,11)
    num_nodes = []
    num_input_nodes = random.randint(5,800)

    for i in range(num_layers):
        num_nodes.append(random.randint(1,800))

    start_time = time.time()
    model = bulk_NN(num_nodes, num_layers, num_input_nodes)

    model.fit(x_trn, dummy_y_trn, epochs=epochs_num, batch_size=batch_num, verbose=0)

    predictions = predict(model,x_tst)
    x = testAccuracy(predictions,y_tst)

    runtime = time.time() - start_time

    r_time = open("bulk_runtimes.tsv", "a")
    r_time_filesize = os.path.getsize("bulk_runtimes.tsv")
    if(r_time_filesize > 0):
        r_time.write("\n")
    r_time.write("%f" % runtime)
    r_time.close()

    bulk_epoch = open("num_epochs.tsv", "a")
    l_bulk_filesize = os.path.getsize("num_epochs.tsv")
    if(l_bulk_filesize > 0):
        bulk_epoch.write("\n")
    bulk_epoch.write("%.0f" % epochs_num)
    bulk_epoch.close()

    bulk_batch = open("num_batch.tsv", "a")
    b_bulk_filesize = os.path.getsize("num_batch.tsv")
    if(b_bulk_filesize > 0):
        bulk_batch.write("\n")
    bulk_batch.write("%.0f" % batch_num)
    bulk_batch.close()

    bulk_accuracy = open("bulkNN_accuracies.tsv", "a")
    b_a_filesize = os.path.getsize("bulkNN_accuracies.tsv")
    if(b_a_filesize > 0):
        bulk_accuracy.write("\n")
    bulk_accuracy.write("%.2f" % x)
    bulk_accuracy.close()

    bulk_layers = open("bulk_layers.tsv", "a")
    b_l_filesize = os.path.getsize("bulk_layers.tsv")
    if(b_l_filesize > 0):
        bulk_layers.write("\n")
    bulk_layers.write("%.0f" % num_layers)
    bulk_layers.close()

    bulk_nodes = open("bulk_nodes.tsv", "a")
    b_n_filesize = os.path.getsize("bulk_nodes.tsv")
    if(b_n_filesize > 0):
        bulk_nodes.write("\n")
    for i in range(len(num_nodes)):
        bulk_nodes.write("%.0f" % num_nodes[i])
        bulk_nodes.write(" ")
    bulk_nodes.close()

    input_nodes = open("input_nodes.tsv", "a")
    b_i_filesize = os.path.getsize("input_nodes.tsv")
    if(b_i_filesize > 0):
        input_nodes.write("\n")
    input_nodes.write("%.0f" % num_input_nodes)
    input_nodes.close()

    time.sleep(1)
