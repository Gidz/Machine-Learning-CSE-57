'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
from scipy.optimize import minimize
import pickle
from math import sqrt
import time

def one_heater(label):
    heat = np.zeros((2,))
    heat[label] = 1
    return heat

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1.0/(1.0+np.exp(-1.0*z))# your code here

def feature_selection(data, threshold):
    variances = np.var(data, axis=0, ddof=True)
    select = np.where(variances > threshold)
    return select[0]

def transfer_derivative(z):
    return (1.0 - z) * z

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W


# Replace this with your nnObjFunction implementation
def like_cost(y, p):
    return - np.sum(np.multiply(y, np.log(p)) + np.multiply((1 - y), np.log(1 - p)))


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, the training data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, train_data, train_label, train_hot, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    num_train = train_data.shape[0]
    train_data_bias = np.column_stack([train_data, np.ones(num_train, dtype=np.float64)])

    h1 = sigmoid(np.dot(train_data_bias, w1.transpose()))
    h1 = np.column_stack([h1, np.ones(h1.shape[0], dtype=np.float64)])
    o1 = sigmoid(np.dot(h1, w2.transpose()))

    delt_out = train_hot - o1


    J1 = like_cost(train_hot, o1)
    obj_val = J1 + (lambdaval / (2.0 * num_train))

    # This is vector calculation


    grad_w2 = np.add(np.dot(-(delt_out * transfer_derivative(o1)).T, h1), lambdaval * w2) * (1.0 / num_train)
    sum= np.dot(delt_out * transfer_derivative(o1), w2)
    hidmat = np.multiply(-(1 - h1), h1)
    result = np.multiply(hidmat, sum)[:, :-1]
    res1 = np.dot(result.transpose(), train_data_bias)
    grad_w1 = (res1 + (lambdaval * w1)) / float(num_train)
    obj_grad = np.concatenate((np.array(grad_w1).flatten(), np.array(grad_w2).flatten()), 0)
    # print('V')
    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = []
    data_bias = np.column_stack([data, np.ones(data.shape[0])])
    o1 = sigmoid(np.dot(data_bias, w1.T))
    o1_bias = np.column_stack([o1, np.ones(o1.shape[0])])
    o2 = sigmoid(np.dot(o1_bias, w2.T))
    for i in range(data.shape[0]):
        labels.append(np.argmax(o2[i]))

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]

    # one-hot
    train_hot = np.array([one_heater(int(i)) for i in train_y])
    test_hot = np.array([one_heater(int(j)) for j in test_y])
    vali_hot = np.array([one_heater(int(k)) for k in valid_y])

    print("Preprocessing done!!")
    return train_x, train_y, valid_x, valid_y, test_x, test_y, train_hot, features

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label, train_hot, features = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, train_hot, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' :50}    # Preferred value.

clock_start = time.time();
print("Clock Started")
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
clock_end = time.time()
print("Learned in ",str(clock_end-clock_start), " seconds")
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

obj = [features, n_hidden, w1, w2, lambdaval]
pickle.dump(obj, open('params.pickle', 'wb'))
