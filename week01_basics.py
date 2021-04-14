"""warm up excercises with numpy and practice with perceptron classifiers"""

import numpy as np

#create array
A = np.array([[1,2,3],[4,5,6]])

#transpose
def tp(A):
    return np.transpose(A)

#row vector
def rv(value_list):
    return np.array([value_list])

#column vector
def cv(value_list):
    return rv(value_list).T

#length
def length(col_v):
    return np.sum(col_v * col_v)**0.5

#normalize
def normalize(col_v):
    return col_v/length(col_v)

#last column
def index_final_col(A):
    return A[:,-1:]

#data
data = np.array([[150,5.8],[130,5.5],[120,5.3]])

#matrix multiplication
def transform(data):
    return np.dot(data,np.array([[1], [1]]))

#signed distance
def signed_dist(x, th, th0):
    return (np.dot(np.transpose(th),x)+th0)/np.sum(th*th)**0.5

#side of hyperplane
def positive(x, th, th0):
    return np.sign(signed_dist(x, th, th0))

#compute score
def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)

#best separator
def best_separator(data, labels, ths, th0s):
    index = np.argmax(score(data, labels, ths, th0s))
    return (ths[:,[index]],th0s[:,[index]])