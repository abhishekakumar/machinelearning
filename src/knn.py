from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from itertools import permutations
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import datasets as ds
import pylab

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

misclassification_error = []

class knn:
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

		#Get the number of classes and the class labels
		self.class_list = np.unique(self.y_train)
		self.num_of_classes = len(self.class_list)

	def classify_knn(self):
		self.knn = KNeighborsClassifier(n_neighbors=self.num_neighbors)
		self.knn.fit(self.x_train, self.y_train)
		self.result = self.knn.predict(self.x_test)
		

	def plot_cross_validation(self):
		k_fold_values = {'breast_cancer': 10, 'digits': 10, 'forest_mapping': 6}
		class_scores = []

		for n in range(1, self.num_of_classes + 1):
			knn = KNeighborsClassifier(n_neighbors=n)
			self.y_train = np.reshape(self.y_train, (len(self.y_train),))
			scores = cross_val_score(knn, self.x_train, self.y_train, cv=k_fold_values[self.dataset], scoring='accuracy')
			class_scores.append(scores.mean())

		#Return the number of neighbors corresponding to the max class_score
		self.num_neighbors = class_scores.index(max(class_scores)) + 1
	
	    # plot number of Neighbors vs accuracy
		fig1 = plt.figure(figsize=(8, 6), dpi=80).add_subplot(111)
		fig1.plot(np.arange(1, self.num_of_classes + 1), class_scores)
		fig1.set_xlabel('Number of Neighbors')
		fig1.set_ylabel('Mean Accuracy Score')
		fig1.set_title('KNN - neighbors vs accuracy', fontsize=12)
		pylab.show()

	def checkValidation(self):
		error = 0.0
		for i in range(len(self.result)):
			if self.result[i] != self.y_test[i]:
				error += 1
		misclassification_error.append(error/len(self.y_test))



def test_knn(dataset):  
	x_data, y_data = ds.retrieve_data_sets(dataset)
	k_fold_values = {'breast_cancer': 10, 'digits': 10, 'forest_mapping': 6}
	train_percentage = [0.1, 0.2, 0.3, 0.4, 0.5]


	# Iterate through the training percentages
	for train_percent in train_percentage:       
		learn_knn = knn(dataset, train_percent)
		learn_knn.plot_cross_validation()
		learn_knn.classify_knn()
		learn_knn.checkValidation()

    # plot misclassification error against training percentages
	fig1 = plt.figure(figsize=(8, 6), dpi=80).add_subplot(111)
	fig1.plot(train_percentage, misclassification_error)
	fig1.set_xlabel('Training Percent')
	fig1.set_ylabel('Misclassification Error')
	fig1.set_title('KNN - Misclassification Error', fontsize=12)
	pylab.show()
