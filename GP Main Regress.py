"""
=====================================================
Gaussian process regression (GPR)
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
Ntrain = np.int(900)
Ntest = np.int(250)

Xtrain = X[0:Ntrain-1,:]
ytrain = y[0:Ntrain-1]
Xtest = X[N-Ntest:N,:]
ytest = y[N-Ntest:N]


#create one-hot vector for the targets. i.e. the labels we want to correctly classify
yhotvec = []
for n in range(0, len(y)):

    zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    yvalue = y[n]
    zeros[yvalue] += 1
    yhotvec.append(zeros)


yhotvectrain = yhotvec[0:Ntrain-1]
yhotvectest = yhotvec[N-Ntest:N]

#kernel = 1.0 * RBF(1.0) #isotropic
kernel = DotProduct(5.0)
#kernel = Matern(0.1)
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
# plt.figure(1)
# plt.plot(tsize, radial, 'r')
# plt.plot(tsize, dot)
# plt.plot(tsize, matern, 'k--')
# plt.xlabel('Training Data Size')
# plt.ylabel('Test Error Rate (%)')
# plt.legend(labels = ['Radial', 'Dot Product', 'Materné'])
# plt.xlim([500, 1500])
# plt.title('GPR Test Error Rate (L = 1, Test Size = 100)')
# plt.yticks(np.arange(0, 30, step = 5))
# plt.grid()
#
#
#
# tsize = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
# radial = [9, 8, 6, 6, 6.8, 6.8, 7.2, 7.2, 6.8, 6, 6]
# dot = [19.6, 21.6, 23.2, 20.8, 21.6, 20.4, 23.6, 22, 23.6, 26.8, 31.2]
# matern = [9, 8, 6, 6, 6.8, 6.8, 7.2, 7.2, 6.8, 6, 6]
# plt.figure(2)
# plt.plot(tsize, radial, 'r')
# plt.plot(tsize, dot)
# plt.plot(tsize, matern, 'k--')
# plt.xlabel('Training Data Size')
# plt.ylabel('Test Error Rate (%)')
# plt.legend(labels = ['Radial', 'Dot Product', 'Materné'])
# plt.xlim([500, 1500])
# plt.title('GPR Test Error Rate (L = 1, Test Size = 250)')
# plt.yticks(np.arange(0, 30, step = 5))
# plt.grid()
# plt.show()
