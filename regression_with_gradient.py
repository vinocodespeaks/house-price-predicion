import numpy as np
import matplotlib.pyplot as plt

#read the data 

data = np.genfromtxt('kc_house_data.csv',delimiter=",") # read the data
data =np.delete(data,0,axis=0)
X = data[:, 19].reshape(-1,1) # -1 tells numpy to figure out the dimension by itself
ones = np.ones([X.shape[0], 1]) # create a array containing only ones 
X = np.concatenate([ones, X],1) # cocatenate the ones to X matrix
y = data[:, 2].reshape(-1,1) # create the y matrix

plt.scatter(data[:, 19].reshape(-1,1), y)

# setting learning rates and epoches
alpha = 0.0001
iters = 1000
#initialize hypter parameter
# theta = thera0 +theta1*X
theta = np.array([[1.0, 1.0]])

def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2) # @ means matrix multiplication of arrays. If we want to use * for multiplication we will have to convert all arrays to matrices
    return np.sum(inner) / ( 2*len(X))

def gradientDescent(X, y, theta, alpha, iters):
    for i in range(1000):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost = computeCost(X, y, theta)
        if i % 10 == 0: # just look at cost every ten loops for debugging
           print(cost)
    return (theta, cost)
g, cost = gradientDescent(X, y, theta, alpha, iters)  
print(g, cost)

