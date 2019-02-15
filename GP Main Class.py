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
Ntrain = np.int(800)
Ntest = np.int(250)

Xtrain = X[0:Ntrain-1,:]
ytrain = y[0:Ntrain-1]
Xtest = X[N-Ntest:N,:]
ytest = y[N-Ntest:N]


#kernel = 1.0 * RBF([1.0]) #isotropic kernel
#kernel = DotProduct(1.0)
kernel = Matern(0.5)
gpc_rbf = GaussianProcessClassifier(kernel=kernel).fit(Xtrain, ytrain)
yp_train = gpc_rbf.predict(Xtrain)
train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
yp_test = gpc_rbf.predict(Xtest)
test_error_rate = np.mean(np.not_equal(yp_test,ytest))
#print('Training error rate')
#print(train_error_rate)
print('Test error rate')
print(test_error_rate)

#testing set 100
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
# plt.ylim([0, 100])
# plt.yticks(np.arange(0, 100, step = 10))
# plt.title('Test Error (L = 1, Test Size = 100)')
# plt.grid()
#
# #testing set 250
# radial = [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
# dot = [11.6, 13.6, 12.8, 12.8, 15, 13.6, 14, 14.8, 13.6, 13.2, 13.6]
# matern = [32.4, 31.2, 28.8, 26.4, 24.8, 23.2, 22.4, 22, 20.8, 20.4, 20.4]
#
# plt.figure(3)
# plt.plot(tsize, radial)
# plt.plot(tsize, dot)
# plt.plot(tsize, matern)
# plt.xlabel('Training Set Size')
# plt.ylabel('Test Error (%)')
# plt.legend(labels = ['Radial', 'Dot Product', 'Materné'])
# plt.xlim([500, 1500])
# plt.ylim([0, 100])
# plt.yticks(np.arange(0, 100, step = 10))
# plt.title('Test Error (L = 1, Test Size = 250)')
# plt.grid()
# plt.show()
#

