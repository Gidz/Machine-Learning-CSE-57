import numpy as np
from scipy.optimize import minimize

from numpy.linalg import det, inv
from math import sqrt, pi

import matplotlib.pyplot as plt
import pickle
import sys
import math


def ldaLearn(X, y):
    y = y.flatten()

    classes = np.unique(y)
    k = int(np.max(y))
    d = X.shape[1]
    means = np.zeros((d, k))
    for i in range(k):
        means[:, i] = np.mean(X[y == classes[i]], 0)
        # print(means.shape)
    covmat = np.cov(np.transpose(X))

    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    y = y.flatten()

    classes = np.unique(y)
    k = int(np.max(y))
    d = X.shape[1]
    means = np.empty((d, k))
    covmats = []
    for i in range(k):
        # print(float(i))
        means[:, i] = np.mean(X[y == classes[i]], axis=0)
        covmats.append(np.cov(X[y == classes[i]].T))

    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # p=np.empty(5)
    N = Xtest.shape[0]
    d = means.shape[0]
    k = means.shape[1]

    invCov = np.linalg.inv(covmat)
    detCov = np.linalg.det(covmat)

    denom = np.power(2 * math.pi, d / 2) * np.sqrt(detCov)
    pdf = np.empty((N, k))
    y1pred = []
    aight = 0
    for i in range(N):
        for j in range(k):
            diff = Xtest[i, :] - means[:, j]
            numer = np.exp(-.5 * np.dot(np.dot(diff, invCov).T, (diff)))
            score = numer / denom
            pdf[:, j] = score
        y1pred.append(pdf.argmax() + 1)
        if y1pred[i] == ytest[i]:
            aight += 1
    ypred = np.vstack(y1pred)
    acc = aight / N

    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    d = means.shape[0]
    k = means.shape[1]

    pdf = np.empty((N, k))
    y1pred = []
    aight = 0
    for i in range(N):
        for j in range(k):
            detCov = np.linalg.det(covmats[k - 1])
            invCov = np.linalg.inv(covmats[k - 1])
            denom = np.power(2 * math.pi, d / 2) * np.sqrt(detCov)
            diff = Xtest[i, :] - means[:, j]
            numer = np.exp(-.5 * np.dot(np.dot(diff, invCov).T, (diff)))
            score = numer / denom
            pdf[:, j] = score
        y1pred.append(pdf.argmax() + 1)
        if y1pred[i] == ytest[i]:
            aight += 1
    acc = aight / N
    ypred = np.vstack(y1pred)
    # print(ypred.shape)

    return acc, ypred


def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    w = np.dot((np.linalg.inv(X.T.dot(X))), (X.T.dot(y)))
    return w


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    N = X.shape[0]
    d = X.shape[1]
    w = ((np.linalg.inv(((X.T).dot(X)) + (lambd * np.eye(d)))).dot(X.T)).dot(y)
    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD
    mse = (1.0 / Xtest.shape[0]) * np.sqrt(np.sum(np.square((ytest - np.dot(Xtest, w)))))
    return mse

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    N = X.shape[0]


    #error = ((np.linalg.inv(((X.T).dot(X)) + (lambd * np.eye(d)))).dot(X.T)).dot(y)
    #error = ((np.sum(np.square(y - np.dot(X, w.T)))) / (2)) + ((lambd * np.dot(w.T, w)) / 2)
    wT = np.array([w]).T
    w2 = np.array([w])
    term1=(y - np.dot(X, wT))
    error = ((np.dot(term1.T, term1)) / (2.0 * N)) + ((np.dot(lambd, np.dot(wT.T, wT))) / 2)
    yTx = np.dot(y.T, X)
    xTx = np.dot(X.T, X)
    error_grad = ((((-1.0 * yTx) + (np.dot(w.T, xTx))) / N) + (lambd * w)).flatten()


    return error, error_grad


def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))

    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    Xd = np.zeros((N, p + 1))
    for i in range(p + 1):
        Xd[:, i] = pow(x, i)
    # print("this is XD" + str(Xd.shape))
    return Xd


# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
#print('LDA Accuracy = ' + str(ldaacc))
# QDA
means, covmats = qdaLearn(X, y)
qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
#print('QDA Accuracy = ' + str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
# plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mletr=testOLERegression(w, X, y)
mle = testOLERegression(w, Xtest, ytest)

w_i = learnOLERegression(X_i, y)
mletr_i=testOLERegression(w_i, X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)

print('MSE Train Data without intercept ' + str(mletr))
print('MSE Train Data with intercept ' + str(mletr_i))

print('MSE Test Data without intercept ' + str(mle))
print('MSE Test Data with intercept ' + str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 20}  # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    print(lambd,",",mses4[i][0],",",mses4_train[i][0])
    i = i + 1

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])
plt.show()
# Problem 5
pmax = 7
lambda0=0
lambda_opt = lambdas[np.argmin(mses3)]  # REPLACE THIS WITH lambda_opt estimated from Problem 3
#####for lambda optimum
mses5_train = np.zeros((pmax, 2))
mses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    mses5_train[p, 0] = testOLERegression(w_d1, Xd, y)
    mses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    mses5_train[p, 1] = testOLERegression(w_d2, Xd, y)
    mses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)
print("REGULARISED DATA\n-----------------");
print("P,Lambda,TestingMSE,TrainingMSE")
for i in range(pmax):
    print(i+1,",",lambda_opt,",",mses5[i][0],",",mses5_train[i][0])

print("\nNON-REGULARISED DATA\n--------------------");
print("P,Lambda,TestingMSE,TrainingMSE")
for i in range(pmax):
    print(i+1,",",lambda_opt,",",mses5[i][1],",",mses5_train[i][1])
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train[range(pmax),0])
plt.plot(range(pmax), mses5_train[range(pmax),1])
plt.title('MSE for Train Data')
plt.legend(('No Regularization', 'Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5[range(pmax),0])
plt.plot(range(pmax), mses5[range(pmax),1])
plt.title('MSE for Test Data')
plt.legend(('No Regularization', 'Regularization'))
plt.show()
#######for lambda 0
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    mses5_train[p, 0] = testOLERegression(w_d1, Xd, y)
    mses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lambda0)
    mses5_train[p, 1] = testOLERegression(w_d2, Xd, y)
    mses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)
print("REGULARISED DATA\n-----------------");
print("P,Lambda,TestingMSE,TrainingMSE")
for i in range(pmax):
    print(i+1,",",lambda0,",",mses5[i][0],",",mses5_train[i][0])

print("\nNON-REGULARISED DATA\n--------------------");
print("P,Lambda,TestingMSE,TrainingMSE")
for i in range(pmax):
    print(i+1,",",lambda0,",",mses5[i][1],",",mses5_train[i][1])
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train[range(pmax),0])
plt.plot(range(pmax), mses5_train[range(pmax),1])
plt.title('MSE for Train Data')
plt.legend(('No Regularization', 'Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5[range(pmax),0])
plt.plot(range(pmax), mses5[range(pmax),1])
plt.title('MSE for Test Data')
plt.legend(('No Regularization', 'Regularization'))
plt.show()
