import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import random
import os
import sys
import time

dataframe = pd.read_csv("Vivek_Ranged_Tree_Data.csv",delimiter=",",header=None)
data = dataframe.values

## Training data for leaf networks
x_weights_trn = data[:4500,0]
y_weights_trn = data[:4500,1]
x_pattern_trn = data[:4500,2]
y_pattern_trn = data[:4500,3]
x_coo_trn = data[:4500,4]
y_coo_trn = data[:4500,5]
x_breed_trn = data[:4500,6]
y_breed_trn = data[:4500,7]
x_btype_trn = data[:4500,8]
y_btype_trn = data[:4500,9]

## Training data for root network
x_root_trn = [y_weights_trn, y_pattern_trn, y_coo_trn, y_breed_trn, y_btype_trn]
x_root_trn = np.array(x_root_trn)
x_root_trn = x_root_trn.T
y_root_trn = []
for i in range(len(y_weights_trn)):
    y_root_trn.append(y_weights_trn[i]+y_pattern_trn[i]+y_coo_trn[i]+y_breed_trn[i]+y_btype_trn[i])

## Testing data for leaf networks
x_weights_tst = data[4501:5000,0]
y_weights_tst = data[4501:5000,1]
x_pattern_tst = data[4501:5000,2]
y_pattern_tst = data[4501:5000,3]
x_coo_tst = data[4501:5000,4]
y_coo_tst = data[4501:5000,5]
x_breed_tst = data[4501:5000,6]
y_breed_tst = data[4501:5000,7]
x_btype_tst = data[4501:5000,8]
y_btype_tst = data[4501:5000,9]

## Testing data for root network
x_root_tst = [y_weights_tst, y_pattern_tst, y_coo_tst, y_breed_tst, y_btype_tst]
x_root_tst = np.array(x_root_tst)
x_root_tst = x_root_tst.T
y_root_tst = []
for i in range(len(y_weights_tst)):
    y_root_tst.append(y_weights_tst[i]+y_pattern_tst[i]+y_coo_tst[i]+y_breed_tst[i]+y_btype_tst[i])

## One hot encoding
dummy_y_train_weights = np_utils.to_categorical(y_weights_trn)
dummy_y_train_weights = np.delete(dummy_y_train_weights, obj=0,axis=1)
dummy_y_train_patt = np_utils.to_categorical(y_pattern_trn)
dummy_y_train_patt = np.delete(dummy_y_train_patt, obj=0,axis=1)
dummy_y_train_coo = np_utils.to_categorical(y_coo_trn)
dummy_y_train_coo = np.delete(dummy_y_train_coo, obj=0,axis=1)
dummy_y_train_breed = np_utils.to_categorical(y_breed_trn)
dummy_y_train_breed = np.delete(dummy_y_train_breed, obj=0,axis=1)
dummy_y_train_btype = np_utils.to_categorical(y_btype_trn)
dummy_y_train_btype = np.delete(dummy_y_train_btype, obj=0,axis=1)
dummy_y_train_root = np_utils.to_categorical(y_root_trn)
dummy_y_train_root = np.delete(dummy_y_train_root, obj=0,axis=1)

## Function that creates our neural network for the leafs
def leaf_NN(x_trn, dummy_y_trn, x_tst, num_output, num_epochs, size_of_batch):
    ## Creating model
    model = Sequential()
    ## Adding input layer with rectified linear activation function for input layer
    model.add(Dense(8, input_dim=1, activation='relu'))
    ## Adding output layer with softmax activation function
    model.add(Dense(num_output, activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ## Fitting model
    model.fit(x_trn, dummy_y_trn, epochs=num_epochs, batch_size=size_of_batch, verbose=0)
    ## Getting predictions
    preds = predict(model,x_tst)
    return preds

## Function that creates our root neural network
def root_NN(weights, pattern, coo, breeds, btype, x_trn, dummy_y_trn, num_epochs, size_of_batch):
    model = Sequential()
    model.add(Dense(6, input_dim=5, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(27, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_trn, dummy_y_trn, epochs=num_epochs, batch_size=size_of_batch, verbose=0)
    pred_root_trn = [weights, pattern, coo, breeds, btype]
    pred_root_trn = np.array(pred_root_trn)
    pred_root_trn = pred_root_trn.T
    preds_root = predict(model,pred_root_trn)
    #best_cat = max(preds_root)
    #for i in range(len(preds_root)):
        #if(preds_root[i] == best_cat):
            #print("The best cat is cat number: %.0f" % (i))
    return testAccuracy(preds_root,y_root_tst)

## Function for predicting labels using keras predict function
def predict(model,x):
    preds = np.argmax(model.predict(x), axis=-1)
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
    return final
    #return print("Accuracy: %.2f" % (final))

class Node:
    def __init__(self, data):
        self.children = []
        self.data = data

    def insert(self, data):
        self.children.append(Node(data))

for iteration in range(20):
    start_time = time.time()
    #root_epochs = random.randint(90,300)
    #leaf_epochs = random.randint(90,300)
    #root_batch = random.randint(3,100)
    #leaf_batch = random.randint(3,100)
    root = Node(1)
    root.insert(leaf_NN(x_weights_trn, dummy_y_train_weights, x_weights_tst, 5, 249, 60))
    root.insert(leaf_NN(x_pattern_trn, dummy_y_train_patt, x_pattern_tst, 4, 249, 60))
    root.insert(leaf_NN(x_coo_trn, dummy_y_train_coo, x_coo_tst, 5, 249, 60))
    root.insert(leaf_NN(x_breed_trn, dummy_y_train_breed, x_breed_tst, 5, 249, 60))
    root.insert(leaf_NN(x_btype_trn, dummy_y_train_btype, x_btype_tst, 8, 249, 60))

    root.data = root_NN(root.children[0].data, root.children[1].data, root.children[2].data, root.children[3].data, root.children[4].data, x_root_trn, dummy_y_train_root, 202, 59)

    runtime = time.time() - start_time

    r_time = open("tree_runtimes_perf.tsv", "a")
    r_time_filesize = os.path.getsize("tree_runtimes_perf.tsv")
    if(r_time_filesize > 0):
        r_time.write("\n")
    r_time.write("%f" % runtime)
    r_time.close()

    accuracy = open("accuracies_perf.tsv", "a")
    a_filesize = os.path.getsize("accuracies_perf.tsv")
    if(a_filesize > 0):
        accuracy.write("\n")
    accuracy.write("%.2f" % root.data)
    accuracy.close()
