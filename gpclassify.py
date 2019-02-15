"""
=====================================================
Gaussian process classification (GPC) 
=====================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern

# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)

N,d = X.shape

N = np.int(1797)
Ntrain = np.int(100)
Ntest = np.int(100)


Xtrain = X[0:Ntrain-1,:]
ytrain = y[0:Ntrain-1]
Xtest = X[N-100:N,:]
ytest = y[N-100:N]

print(ytrain)

#create one-hot vector for the targets. i.e. the labels we want to correctly classify
yhotvec = []
for n in range(0, len(y)):

    zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    yvalue = y[n]
    zeros[yvalue] += 1
    yhotvec.append(zeros)

print(yhotvec)
print(len(yhotvec))

yhotvectrain = yhotvec[0:Ntrain-1]
yhotvectest = yhotvec[N-100:N]

kernel = 1.0 * RBF(1.0) #isotropic
#kernel = DotProduct(1.0)
#kernel = Matern(1.0)
gpc_rbr = GaussianProcessRegressor(kernel = kernel, normalize_y = False).fit(Xtrain, yhotvectrain)
yp_train = gpc_rbr.predict(Xtrain)
train_error_rate = np.mean(np.not_equal(yp_train, yhotvectrain))
yp_test = gpc_rbr.predict(Xtest)
yp_test = np.argmax(yp_test, axis = 1)
yhotvectest = np.argmax(yhotvectest, axis = 1)
test_error_rate = np.mean(np.not_equal(yp_test, yhotvectest))
print('Training error rate')
print(train_error_rate)
print('Test error rate')
print(test_error_rate)

# tsize = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
# radial = [7, 6, 3, 2, 3, 3, 4, 4, 3, 3, 3]
# dot = [14, 15, 18, 16, 15, 14, 19, 19, 18, 19, 24]
# matern = [7, 6, 3, 2, 3, 3, 4, 4, 3, 3, 3]
# plt.plot(tsize, radial, 'r')
# plt.plot(tsize, dot)
# plt.plot(tsize, matern, 'k--')
# plt.xlabel('Training Data Size')
# plt.ylabel('Test Error Rate (%)')
# plt.legend(labels = ['Radial', 'Dot Product', 'Materné'])
# plt.xlim([500, 1500])
# plt.title('GP Regression Classifier Test Error Rate (l = 1)')
# plt.show()



# #kernel = 1.0 * RBF([3.0]) #isotropic kernel
# kernel = DotProduct(1.0) #dotproduct kernel
# #kernel = Matern(1.0) #materné kernel
#
# gpc_rbf = GaussianProcessClassifier(kernel=kernel).fit(Xtrain, ytrain)
# yp_train = gpc_rbf.predict(Xtrain)
# train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
# yp_test = gpc_rbf.predict(Xtest)
# test_error_rate = np.mean(np.not_equal(yp_test,ytest))
# #print('Training error rate')
# #print(train_error_rate)
# print('Test error rate')
# print(test_error_rate)

# tsize = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
# radial = [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
# dot = [7, 11, 8, 8, 7, 9, 8, 8, 7, 8, 7]
# matern = [32, 31, 27, 25, 22, 19, 19, 18, 17, 17, 17]
#
# plt.figure(2)
# plt.plot(tsize, radial)
# plt.plot(tsize, dot)
# plt.plot(tsize, matern)
# plt.xlabel('Training Data Size')
# plt.ylabel('Test Error Rate (%)')
# plt.legend(labels = ['Radial', 'Dot Product', 'Materné'])
# plt.xlim([500, 1500])
# plt.title('Test Error with various covariance functions (l = 1)')
# plt.show()

# kernel = 1.0 * RBF([1.0]) #isotropic kernel
# #kernel = DotProduct(1.0) #dotproduct kernel
# #kernel = Matern(1.0) #materné kernel
#
# gpr_rbf = GaussianProcessRegressor(kernel = kernel).fit(Xtrain, ytrain)
# yp_train = gpr_rbf.predict(Xtrain)
# train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
# yp_test = gpr_rbf.predict(Xtest)
# test_error_rate = np.mean(np.not_equal(yp_test,ytest))
#
# print('Training error rate')
# print(train_error_rate)
# print('Test error rate')
# print(test_error_rate)

