import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# See this website for a detailed use on the memory profiler:
# https://pypi.org/project/memory-profiler/

# Below is the import statement for the time measurement:
import time

start_time = time.time()

dataframe = pd.read_csv("Vivek_Ranged_Data.csv", delimiter=",", header=None)
data = dataframe.values

# Loading the data on which the neural network will be trained.
x_trn = data[:9000,0:5]
y_trn = data[:9000,5]
x_tst = data[9001:10000,0:5]
y_tst = data[9001:10000,5]

# One hot encoding
dummy_y_trn = np_utils.to_categorical(y_trn)
dummy_y_trn = np.delete(dummy_y_trn, obj=0, axis=1)

# Function that creates our neural network
@profile
def bulk_NN():
    # Creating model
    model = Sequential()

    # Adding the input layer, using the sigmoid actiation function.
    model.add(Dense(753, input_dim=5, activation='sigmoid'))

    # Adding 7 hidden layers, all using the sigmoid activation function.
    model.add(Dense(336, activation='sigmoid'))
    model.add(Dense(184, activation='sigmoid'))
    model.add(Dense(377, activation='sigmoid'))
    model.add(Dense(466, activation='sigmoid'))
    model.add(Dense(143, activation='sigmoid'))
    model.add(Dense(519, activation='sigmoid'))
    model.add(Dense(654, activation='sigmoid'))

    # Adding the output layer, using the softmax activation function. The number of units must be 27 here.
    model.add(Dense(27, activation='softmax'))

    # Compile the model, using the 'adam' optimizer.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function for predicting labels using keras predict function
@profile
def predict(model,x):
    preds = np.argmax(model.predict(x), axis=-1)
    for i in range(len(preds)):
        preds[i] = preds[i] + 1
    return preds

# Function for testing accuracy of predictions
@profile
def testAccuracy(preds, labels):
    count = 0
    for i in range(len(preds)):
        if(preds[i] == labels[i]):
            count += 1
    final = 100*(count/len(labels))
    return print("Accuracy: %.2f" % (final), "\n")

model = bulk_NN()

# Fitting our model with the training data
model.fit(x_trn, dummy_y_trn, epochs=197, batch_size=79, verbose=0)

# Testing accuracy after training
predictions = predict(model,x_tst)
testAccuracy(predictions,y_tst)

print("Total elapsed time for the program to run:", time.time() - start_time, "seconds\n\n\n")