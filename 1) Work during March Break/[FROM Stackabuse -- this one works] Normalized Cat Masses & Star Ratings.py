'''
FROM: https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/

This works!!

The "feature_set" values are the first 5 cat masses from our Excel spreadsheet,
that had to be normalized (i.e. between 0 and 1). The reason for this is
because the Sigmoid function (our activation function) maps all values
between -1 and 1 (read the link above, it explains it nicely!).

So, to normalize, I divided all the cat masses by 30 (the highest cat mass
in our Excel spreadsheet.)

I ALSO had to normalize the star ratings (for the same reason). I divided all
the star ratings in our Excel spreadsheet by 5 (the highest star rating).

Then, when I run the code below, we get a really good result for the
expected star rating when we give the NN a particular cat mass! Nice!

[SEE THE EXCEL SPREADSHEET TITLED "[FROM Stackabuse -- this one works] Normalized Cat Masses & Star Ratings.xlsx"]
'''

import numpy as np
# INPUT NORMALIZED CAT MASS DATA HERE!
feature_set = np.array([[0.717494106],[0.277365855],[0.237134566],
                        [0.772217553],[0.652803061]])
# INPUT NORMALIZED CAT STAR RATING HERE!
labels = np.array([[0.4,0.8,0.8,0.4,0.6]])
labels = labels.reshape(5,1)

np.random.seed(42)
weights = np.random.rand(1,1)
bias = np.random.rand(1)
lr = 0.05

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

for epoch in range(15000):
    inputs = feature_set

    # feedforward step1
    XW = np.dot(feature_set, weights) + bias

    #feedforward step2
    z = sigmoid(XW)


    # backpropagation step 1
    error = z - labels

    #print(error.sum())

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num

single_point = np.array([0.772216667]) # INPUT NORMALIZED CAT MASS HERE!
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)