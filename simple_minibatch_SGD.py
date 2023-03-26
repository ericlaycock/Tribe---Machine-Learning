import pandas as pd
import numpy as np
import math

feature = [1,2,3,4]
label = [2,3,4,5]

training_data = np.array([[x,y] for x,y in zip(feature,label)])

epochs = 2
learning_rate = 0.1
mbatches = 2

def getMiniBatches(data,m):
    arr = []
    loops_left = math.ceil(len(data)/m)
    start = 0
    end = m

    while(loops_left):
        if(len(data[start:end])<m):
            arr.append(data[start:])
            break
        arr.append(data[start:end])
        start += m
        end += m
        loops_left -=1
    return arr

w = np.random.random()

#n^3 runtime
# for e in range(epochs):
#     np.random.shuffle(training_data)
#     mini_batches = getMiniBatches(training_data.tolist(),mbatches)
#     for batch in mini_batches:
#         MSEs = []
#         gradients = []
#         for example in batch:
#             prediction = w*example[0]
#             print("prediction: %s, actual: %s"%(prediction,example[1]))
#             mse = (example[1]-prediction)**2
#             MSEs.append(mse)
#             gradient = -1 * example[0] * (example[1] - prediction)
#             gradients.append(gradient)
#         meanMSE = np.mean(np.array(MSEs))
#         meanGradient = np.mean(np.array(gradients))
#         w = w-learning_rate*meanGradient

# n^2 runtime - use dot product to generate vector of predictions instead of manually executing
for e in range(epochs):
    np.random.shuffle(training_data)
    mini_batches = getMiniBatches(training_data.tolist(),mbatches)
    for batch in mini_batches:
        predictions = np.dot(batch[:,0], w)
        errors = predictions - batch[:,1]
        meanMSE = np.mean(errors**2)
        meanGradient = np.mean(-2 * batch[:,0] * errors)
        w -= learning_rate * meanGradient



