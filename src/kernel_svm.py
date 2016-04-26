import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from numpy import linalg as LA
import math

####
#
# Two-class kernel SVM.
#
####
#
#
# N is the number of training patterns.
# K is the amount of data points in each pattern.
# C is the regularizer constant.
# Kernel is passed in as a function with signature 
# 	`def kernel(x, y): return double' where x and y are
# 	numpy.array row vectors detailing a pattern.
#
# To classify a pattern simply call kernel_svm.classify(pattern)
#
#
####
class kernel_svm:

	# for support vector rejection
	eps = 0.0001

	###
	# MAKE SURE EVERYTHING GIVEN IS numpy.array....
	###
	#
	#
	###
	# if there are N patterns each with dimensionality K
	# train_x is an NxK numpy.array
	# train_y is an Nx1 numpy.array where
	# train_y[i] = 1 if pattern i is in class 1, -1 otherwise
	#
	# C = outlier/regularizer constant
	###
	#
	#
	###
	# Kernel is the kernel function passed in...
	# def kernel(x, y): returns double
	# where
	# x = numpy row vector (numpy.array)
	# y = numpy row vector (numpy.array)
	###
	def __init__(self, train_x, train_y, N, K, C, kernel):
		self.train_x = train_x
		self.train_y = train_y
		self.N = N
		self.K = K
		self.C = C
		self.kernel = kernel
		self.solve_qp()
		

	def solve_qp(self):
		P = np.zeros( (self.N, self.N) )
		q = np.zeros( (self.N, 1) )
		G = np.zeros( (2*self.N, self.N) )
		h = np.zeros( (2*self.N, 1) )
		A = np.zeros( (1, self.N) )
		b = np.zeros( (1, 1) )
		for i in range(self.N):
			q[i,0] = -1.0
			A[0,i] = self.train_y[i]
			G[i,i] = -1.0
			G[i+self.N,i] = 1.0
			h[i,0] = 0.0
			h[i+self.N,0] = self.C
			for j in range(self.N):
				P[i, j] = self.train_y[i]*self.train_y[j]*self.kernel(self.train_x[i,], self.train_x[j,])

		P = matrix(P)
		q = matrix(q)
		G = matrix(G)
		h = matrix(h)
		A = matrix(A)
		b = matrix(b)

		# suppress dumb warnings
		solvers.options['show_progress'] = False
		sol = solvers.qp(P, q, G, h, A, b)
		lamsol = sol['x']

		thetazero_sum = 0.0
		cnt = 0.0
		for n in range(self.N):
			lamn = lamsol[n, 0]

			if (lamn <= kernel_svm.eps):
				continue

			cnt = cnt + 1
			yn = self.train_y[n]
			rown = self.train_x[n,]
			sum_test = 0.0

			for m in range(self.N):
				lamc = lamsol[m, 0]
				ym = self.train_y[m]
				rowm = self.train_x[m,]
				sum_test = sum_test + lamc*ym*self.kernel(rowm, rown)

			thetazero_sum = thetazero_sum + (1.0/yn - sum_test)


		if (cnt != 0):
			self.thetanot = thetazero_sum / cnt
		else:
			self.thetanot = 0
		self.lamsol = lamsol


	# pattern in a row vector... (np.array)
	# (> 0 => class 1), (< 0 => class 2)
	def classify(self, pattern):
		val = 0

		for m in range(self.N):
			lamc = self.lamsol[m, 0]
			ym = self.train_y[m]
			rowm = self.train_x[m,]
			val = val + lamc*ym*self.kernel(rowm, pattern)

		val = val+self.thetanot

		return val

####
#
# Multi-class kernel SVM.
#
####
#
#
# N is the number of training patterns.
# K is the amount of data points in each pattern.
# M is the number of classes. Classes are labelled [1, M]
# C is the regularizer constant.
# Kernel is passed in as a function with signature 
# 	`def kernel(x, y): return double' where x and y are
# 	numpy.array row vectors detailing a pattern.
#
# To classify a pattern simply call multiclass_svm.classify(pattern)
# the pattern label will be returned
#
#
####
class multiclass_svm:

	###
	### read kernel_svm __init__ for details
	### train_y[i] = k if pattern i's class is `k'
	### k = 0, 1, ..., M-1 (where M is the number of classes)
	###
	def __init__(self, train_x, train_y, N, K, M, C, kernel):
		self.M = M
		self.train_x = train_x
		self.train_y = train_y
		svms = []
		for i in range(M):
			# create SVM for class i
			# (one-versus-all)
			i_train_y = np.copy(train_y)

			# convert to 1 / -1 scheme
			for j in range(N):
				if (i_train_y[j] != float(i)):
					i_train_y[j] = -1.0
				elif (i_train_y[j] == float(i)):
					i_train_y[j] = 1.0

			i_svm = kernel_svm(train_x, i_train_y, N, K, C, kernel)

			svms.append(i_svm)
		self.svms = svms


	# returns an integer between [0, M-1]
	# which is the class that this SVM thinks
	# `pattern' is a part of
	def classify(self, pattern):
		# a very small value
		val_large = -10000.0
		class_large = 0.0
		for i in range(self.M):
			svm = self.svms[i]
			val = svm.classify(pattern)
			if (val > val_large):
				val_large = val
				class_large = i

		return class_large

###
#
# Multi-class kernel SVM estimator for sklearn.
# Backed by multiclass_svm.
#
###
class svm_estimator(BaseEstimator, ClassifierMixin):
	def __init__(self, C, kern):
		self.C = C
		self.kern = kern

	def fit(self, X, y):
		test_y = y.as_matrix()
		N = X.shape[0]
		K = X.shape[1]
		M = 0
		for i in range(N):
			y_val = test_y[i]
			if (y_val > M):
				M = y_val

		self.N = N
		self.K = K
		self.M = M
		self.backing_svm = multiclass_svm(
			X.as_matrix().astype(int), 
			y.as_matrix().astype(int), 
			N, 
			K, 
			M+1, 
			self.C, 
			self.kern,
		)
		return self

	def predict(self, X):
		x_mat = X.as_matrix().astype(int)
		to_classify = x_mat.shape[0]
		out = np.zeros( (to_classify, 1) )
		for i in range(to_classify):
			out[i,0] = self.backing_svm.classify(x_mat[i,])
		return out

###
#
# Gaussian kernel with sigma = 20.0
# (aka, rbf kernel)
#
###
def gauss_kernel(x, y):
	sigma = 20.0
	vec = np.subtract(x, y)
	norm = LA.norm(vec, 2)
	return math.exp(-norm/(2*sigma*sigma))

###
#
# Linear kernel
#
###
def linear_kernel(x, y):
	if (x.shape[0] != y.shape[0]):
		y = y.T
	return np.dot(x, y)

###
#
# Laplacian kernel with gamma = 1.0
#
###
def laplacian_kernel(x, y):
	gamma = 1.0
	vec = np.subtract(x, y)
	norm = LA.norm(vec, 2)
	return math.exp(-norm*gamma)