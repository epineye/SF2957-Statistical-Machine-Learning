import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct

# import some data to play with
digits = datasets.load_digits()
import random

# Load data into train set and test set
digits = datasets.load_digits()
X = digits.data
y = np.array(digits.target, dtype = int)
N,d = X.shape
Ntest = np.int(100)
Ntrain = np.int(1697)
Xtrain = X[0:Ntrain,:]
ytrain = y[0:Ntrain]
Xtest = X[Ntrain:N,:]
ytest = y[Ntrain:N]


def svmsubgradient(Theta, x, y):
#  Returns a subgradient of the objective empirical hinge loss
#
# The inputs are Theta, of size n-by-K, where K is the number of classes,
# x of size n, and y an integer in {0, 1, ..., 9}.
    G = np.zeros(Theta.shape)
    ## IMPLEMENT THE SUBGRADIENT CALCULATION -- YOUR CODE HERE

    return(G)

def sgd(Xtrain, ytrain, maxiter = 10, init_stepsize = 1.0, l2_radius = 10000):
#
# Performs maxiter iterations of projected stochastic gradient descent
# on the data contained in the matrix Xtrain, of size n-by-d, where n
# is the sample size and d is the dimension, and the label vector
# ytrain of integers in {0, 1, ..., 9}. Returns two d-by-10
# classification matrices Theta and mean_Theta, where the first is the final
# point of SGD and the second is the mean of all the iterates of SGD.
#
# Each iteration consists of choosing a random index from n and the
# associated data point in X, taking a subgradient step for the
# multiclass SVM objective, and projecting onto the Euclidean ball
# The stepsize is init_stepsize / sqrt(iteration).
    K = 10
    NN, dd = Xtrain.shape
    print(NN)
    Theta = np.zeros(dd*K)
    Theta.shape = dd,K
    mean_Theta = np.zeros(dd*K)
    mean_Theta.shape = dd,K
    ## YOUR CODE HERE -- IMPLEMENT PROJECTED STOCHASTIC GRADIENT DESCENT

    return Theta, mean_Theta

def Classify(Xdata, Theta):
#
# Takes in an N-by-d data matrix Adata, where d is the dimension and N
# is the sample size, and a classifier X, which is of size d-by-K,
# where K is the number of classes.
#
# Returns a vector of length N consisting of the predicted digits in
# the classes.
    scores = np.matmul(Xdata, Theta)
    inds = np.argmax(scores, axis = 1)
    return(inds)



###subsample_proportion = .2;
###tau = .5;
###[Ktrain, Ktest] = ...
###    GetKernelRepresentation(Atrain, Atest, subsample_proportion, tau);
l2_radius = 40.0
M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
init_stepsize = l2_radius/M_raw
##
###l2_radius_kernel = 60.0;
###M_kernel = sqrt(mean(sum(Ktrain.^2, 2)));
###init_stepsize_kernel = l2_radius_kernel / M_kernel;
###fprintf(1, 'Training SGD\n');
##print('SGD')
maxiter = 40000
Theta, mean_Theta = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius)
#print(Theta)
##X, mean_X = sgd(Atrain, ytrain, maxiter, init_stepsize, l2_radius)
###fprintf(1, 'Training Kernel SGD\n');
###[X_kern, mean_X_kern] = ...
###    sgd(Ktrain, btrain, maxiter, init_stepsize_kernel, l2_radius_kernel);
##Ntest = length(ytest);
##fprintf(1, '** Test error rate for raw data **\n\t');
print('Error rate')
print(np.sum(np.not_equal(Classify(Xtest, mean_Theta),ytest)/Ntest))
##        sum(ClassifyAll(Atest, X_raw) ~= btest) / Ntest);
##fprintf(1, '** Test error rate for kernelized data **\n\t');
##fprintf(1, '%f [last point]\n\t%f [mean of iterates]\n', ...
##        sum(ClassifyAll(Ktest, X_kern) ~= btest) / Ntest, ...
##        sum(ClassifyAll(Ktest, mean_X_kern) ~= btest) / Ntest);
##
##GenerateConfusionMatrix(Atest, btest, X_raw);
##title('Raw data confusion');
##GenerateConfusionMatrix(Ktest, btest, mean_X_kern);
##title('Kernelized data confusion')
