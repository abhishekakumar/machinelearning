from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cluster import KMeans
from itertools import permutations
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import datasets as ds
from sklearn.base import BaseEstimator, ClusterMixin
import pylab

num_clusters = 1

class joshkmeans(BaseEstimator, ClusterMixin):
	def __init__(self):
		self.k_means_back = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10)

	def fit(self, X, y):
		self.k_means_back.fit(X)
		self.result = self.k_means_back.predict(X)
		self.y_train = y

	def find_majority(self, k):
	    myMap = {}
	    maximum = ( '', 0 ) # (occurring element, occurrences)
	    for n in k:
	        if n in myMap: myMap[n] += 1
	        else: myMap[n] = 1

	        # Keep track of maximum on the go
	        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

	    return maximum

	def predict(self, X):
		test = self.k_means_back.predict(X)

		#Maps the cluster labels back to the provided labels by comparing the predict results
		#to the results from training set
		for i in range(len(test)):
			cluster_label = test[i]
			lst = []
			for j in range(len(self.result)):
				if (self.result[j] == cluster_label):
					lst.append(self.y_train[j,0])
			val = self.find_majority(lst)[0]
			test[i] = val
		return test

class k_means:
	def __init__(self, dataset, train_percent):
		print "Training on percent: ", train_percent
		self.dataset = dataset
		self.train_percent = train_percent
		self.separate_data()

	def separate_data(self):
		x_data, y_data = ds.retrieve_data_sets(self.dataset)
		x_data = x_data.as_matrix();
		y_data = y_data.as_matrix();
		x_data[:, 5] = map(float, x_data[:,5])
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
	        x_data,
	        y_data,
	        test_size=1 - self.train_percent,
	        random_state=42
	    )

		self.class_list = np.unique(self.y_train)
		self.num_of_classes = len(self.class_list)

	def classify_kmeans(self):
		k_means = joshkmeans()
		k_means.fit(self.x_train, self.y_train)
		self.result = k_means.predict(self.x_test)

	def plot_cross_validation(self):
		k_fold_values = {'breast_cancer': 10, 'digits': 10, 'forest_mapping': 6}
		cluster_scores = []

		self.x_train = np.matrix(self.x_train)
		self.y_train = np.matrix(self.y_train)
		for n in range(1, len(self.class_list) + 1):
			global num_clusters
			num_clusters = n
			k_means = joshkmeans()
			scores = cross_val_score(k_means, self.x_train, self.y_train, cv=k_fold_values[self.dataset], scoring='accuracy')
			cluster_scores.append(scores.mean())

		num_clusters = cluster_scores.index(max(cluster_scores)) + 1
	    
        # plot number of clusters vs accuracy
		fig1 = plt.figure(figsize=(8, 6), dpi=80).add_subplot(111)
		fig1.plot((self.class_list + 1), cluster_scores)
		fig1.set_xlabel('Number of Clusters')
		fig1.set_ylabel('Mean Accuracy Score')
		fig1.set_title('KMeans - clusters vs accuracy', fontsize=12)
		pylab.show()

	def checkValidation(self):
		error = 0.0
		for i in range(len(self.result)):
			if self.result[i] != self.y_test[i]:
				error += 1
		print "Missclassification Error", (error/len(self.result))*100, "%"


def test_kmeans(dataset): 
	#datasets = ['breast_cancer', 'digits', 'forest_mapping']
	x_data, y_data = ds.retrieve_data_sets(dataset)
	train_percentage = [0.1, 0.2, 0.3, 0.4, 0.5]

	for train_percent in train_percentage:       
		learn_kmeans = k_means(dataset, train_percent)
		learn_kmeans.plot_cross_validation()
		learn_kmeans.classify_kmeans()
		learn_kmeans.checkValidation()