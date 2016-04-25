import kernel_svm
import datasets as ds
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pylab

###
# 
# Driver function for testing multi-class kernel svm
# on the given dataset (given as panda dataframe).
#
###
def test(dataset):
	x_data, y_data = ds.retrieve_data_sets(dataset)
	k_fold_values = {'breast_cancer': 10, 'digits': 10, 'forest_mapping': 6}
	C_values_by_dataset = {'breast_cancer': 1000, 'digits': 1000, 'forest_mapping': 1000}
	train_percentage = [0.1, 0.2, 0.3, 0.4, 0.5]
	training_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	c_values = [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000, 2000]

	# kernel we are using
	kernel = kernel_svm.gauss_kernel

	# Plot c values vs. accuracy, for each train percentage
	for train_percent in train_percentage:
		c_scores = []
		x_train_main, x_test, y_train_main, y_test = train_test_split(
			x_data,
			y_data['class'],
			test_size = 1 - train_percent,
			random_state = 42,
        )

		for c_val in c_values:
			josh_svm = kernel_svm.svm_estimator(c_val, kernel)
			scores = cross_val_score(josh_svm, x_train_main, y_train_main, cv=k_fold_values[dataset],
						scoring='accuracy')
			c_scores.append(scores.mean())

		# plot c-value vs accuracy for this train percent
		svm_fig1 = plt.figure(figsize=(8, 6), dpi=80)
		svm1 = svm_fig1.add_subplot(111)
		svm1.plot(c_values, c_scores, label="C Values")
		svm1.set_xlabel('C Values')
		svm1.set_ylabel('Mean Accuracy Scores')
		svm1.set_xscale('log')
		svm1.set_title('<<CUSTOM SVM>> - C values vs accuracy', fontsize=12)
		pylab.show()

	# Now test with actually chosen C-values
	test_scores = []
	C_choosen = C_values_by_dataset[dataset]
	for train_percent in training_percentage:
		x_train_main, x_test, y_train_main, y_test = train_test_split(
			x_data,
			y_data['class'],
			test_size = 1 - train_percent,
			random_state = 42,
		)

		josh_svm = kernel_svm.svm_estimator(C_choosen, kernel)
		josh_svm.fit(x_train_main, y_train_main)
		predicted_k = josh_svm.predict(x_test)
		scores = metrics.accuracy_score(y_test, predicted_k)
		test_scores.append(scores)

	# plot training data percentage vs accuracy
	svm_fig2 = plt.figure(figsize=(8, 6), dpi=80)
	svm2 = svm_fig2.add_subplot(111)
	svm2.plot(training_percentage, test_scores, label="test percentage")
	svm2.set_xlabel('Training Data Fraction')
	svm2.set_ylabel('Accuracy - Test Set')
	svm2.set_title('<<CUSTOM>> SVM - Accuracy on Test Set', fontsize=12)
	pylab.show()


# Example test. Run in main file :)
if __name__ == "__main__":
	datasets = ['breast_cancer', 'digits', 'forest_mapping']
	kernel_svm_test(datasets[2])