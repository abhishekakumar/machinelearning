from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

import datasets as ds
import plotgraph as pg

train_percentage = [0.1, 0.2, 0.3, 0.4, 0.5]
training_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # For plotting the training accuracy
c_values = [0.001, 0.01, 0.1, 1, 10, 50, 100, 1000]

C_values_by_dataset = {'breast_cancer': 100, 'digits': 0.01, 'forest_mapping': 1000}
kernels_by_dataset = {'breast_cancer': 'rbf', 'digits': 'linear', 'forest_mapping': 'rbf'}
k_fold_values = {'breast_cancer': 10, 'digits': 10, 'forest_mapping': 6}


def SVM(current_dataset):
    x_data, y_data = ds.retrieve_data_sets(current_dataset)

    for train_percent in train_percentage:
        print "Training on percent: ", train_percent
        c_scores = []
        x_train_main, x_test, y_train_main, y_test = train_test_split(
            x_data,
            y_data['class'],
            test_size=1 - train_percent,
            random_state=42
        )

        for c_val in c_values:
            svm_k = svm.SVC(C=c_val, kernel=kernels_by_dataset[current_dataset])
            scores = cross_val_score(svm_k, x_train_main, y_train_main, cv=k_fold_values[current_dataset],
                                     scoring='accuracy')
            c_scores.append(scores.mean())

    # remove if not running in a iPython Notebook
    # %matplotlib inline
    # pg.plot(
    #     c_values,
    #     c_scores,
    #     'C values',
    #     'Mean Accuracy Scores',
    #     'SVM - C values vs accuracy - dataset : ' + current_dataset,
    #     'log'
    # )

    # Check the accuracy on Test Dataset

    test_scores = []
    C_chosen = C_values_by_dataset[current_dataset]
    for train_percent in training_percentage:
        # use the same random state
        x_train_main, x_test, y_train_main, y_test = train_test_split(
            x_data,
            y_data['class'],
            test_size=1 - train_percent,
            random_state=42
        )

        svm_k = svm.SVC(C=C_chosen, kernel=kernels_by_dataset[current_dataset])
        svm_k.fit(x_train_main, y_train_main)
        predicted_k = svm_k.predict(x_test)
        scores = metrics.accuracy_score(y_test, predicted_k)
        test_scores.append(scores)

    # print 'SVM : Data Set : ' + current_dataset + " Accuracy : " + str(test_scores[4])

    # pg.plot(
    #     training_percentage,
    #     test_scores,
    #     'Training Data Fraction',
    #     'Accuracy - Test Set',
    #     'SVM - Accuracy on Test Set - dataset : ' + current_dataset,
    #     'linear'
    # )
    return test_scores