from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import datasets as ds
import deeplearning as dl
import pylab
import kernel_svm_test

import warnings
warnings.filterwarnings("ignore")


def main():
    datasets = ['breast_cancer', 'digits', 'forest_mapping']
    for current_dataset in datasets:
        classify(current_dataset);
    # classify('digits')

def classify(current_dataset):
    print current_dataset
    k_fold_values = {'breast_cancer': 10, 'digits': 10, 'forest_mapping': 6}

    # Set the values below from Cross Validation Graphs
    C_values_by_dataset = {'breast_cancer': 1000, 'digits': 1000, 'forest_mapping': 1000}
    estimator_values_by_dataset = {'breast_cancer': 50, 'digits': 1000, 'forest_mapping': 50}
    max_depth_values_by_dataset = {'breast_cancer': 10, 'digits': 50, 'forest_mapping': 10}


    #current_dataset = datasets[2]  # change accordingly
    x_data, y_data = ds.retrieve_data_sets(current_dataset)

    train_percentage = [0.1, 0.2, 0.3, 0.4, 0.5]
    training_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # For plotting the training accuracy
    c_values = [0.01, 0.1, 1, 10, 50, 100, 1000]

    ##############################
    ### Support Vector Machine ###
    ##############################

    print 'SVM'
    for train_percent in train_percentage:
        c_scores = []
        #     print "\nTraining Set Size : " + str(train_percent*100) + "%"
        x_train_main, x_test, y_train_main, y_test = train_test_split(
            x_data,
            y_data['class'],
            test_size=1 - train_percent,
            random_state=42
        )

        for c_val in c_values:
            svm_k = svm.SVC(C=c_val)
            scores = cross_val_score(svm_k, x_train_main, y_train_main, cv=k_fold_values[current_dataset],
                                     scoring='accuracy')
            c_scores.append(scores.mean())
    # print c_scores

    # remove if not running in a iPython Notebook
    # %matplotlib inline
    svm_fig1 = plt.figure(figsize=(8, 6), dpi=80)
    svm1 = svm_fig1.add_subplot(111)
    svm1.plot(c_values, c_scores, label="C Values")
    svm1.set_xlabel('C Values')
    svm1.set_ylabel('Mean Accuracy Scores')
    svm1.set_xscale('log')
    svm1.set_title('SVM - C values vs accuracy', fontsize=12)
    pylab.show()

    # Check the accuracy on Test Dataset

    test_scores = []
    C_choosen = C_values_by_dataset[current_dataset]
    for train_percent in training_percentage:
        #     print "\nTraining Set Size : " + str(train_percent*100) + "%"
        # use the same random state
        x_train_main, x_test, y_train_main, y_test = train_test_split(
            x_data,
            y_data['class'],
            test_size=1 - train_percent,
            random_state=42
        )

        svm_k = svm.SVC(C=C_choosen)
        svm_k.fit(x_train_main, y_train_main)
        predicted_k = svm_k.predict(x_test)
        scores = metrics.accuracy_score(y_test, predicted_k)
        test_scores.append(scores)

    svm_fig2 = plt.figure(figsize=(8, 6), dpi=80)
    svm2 = svm_fig2.add_subplot(111)
    svm2.plot(training_percentage, test_scores, label="test percentage")
    svm2.set_xlabel('Training Data Fraction')
    svm2.set_ylabel('Accuracy - Test Set')
    svm2.set_title('SVM - Accuracy on Test Set', fontsize=12)
    pylab.show()

    #############################
    ###### Random Forests #######
    #############################

    estimators = [10, 20, 50, 100, 500, 1000]
    max_depths = [5, 10, 20, 50, 100, 200]

    print '\nRandom Forests'

    for train_percent in train_percentage:
        #     print "Training Set Size : " + str(train_percent*100) + "%"
        depth_scores = []
        estimator_scores = []
        #     print "\nTraining Set Size : " + str(train_percent*100) + "%"
        x_train_main, x_test, y_train_main, y_test = train_test_split(
            x_data,
            y_data['class'],
            test_size=1 - train_percent,
            random_state=42
        )

        #     print '\nMax Depths : ' + str(max_depths)
        for max_depth in max_depths:
            rfc_k = RandomForestClassifier(max_depth=max_depth, n_estimators=500, max_features=1)
            scores = cross_val_score(rfc_k, x_train_main, y_train_main, cv=k_fold_values[current_dataset],
                                     scoring='accuracy')
            #         estimator_scores.append(scores.mean())
            depth_scores.append(scores.mean())
        # print depth_scores
        #     train_scores.append(np.array(depth_scores).mean())

        #     print '\nEstimators : ' + str(estimators)
        for estimator in estimators:
            rfc_k = RandomForestClassifier(max_depth=100, n_estimators=estimator, max_features=1)
            scores = cross_val_score(rfc_k, x_train_main, y_train_main, cv=k_fold_values[current_dataset],
                                     scoring='accuracy')
            estimator_scores.append(scores.mean())
    # print estimator_scores

    fig1 = plt.figure(figsize=(8, 6), dpi=80)
    ax1 = fig1.add_subplot(111)
    ax1.plot(estimators, estimator_scores, label="Estimators")
    ax1.set_xlabel('Estimator Values')
    ax1.set_ylabel('Mean Accuracy Scores')
    ax1.set_title('Random Forests - estimator vs accuracy', fontsize=12)
    pylab.show()

    fig2 = plt.figure(figsize=(8, 6), dpi=80)
    ax2 = fig2.add_subplot(111)
    ax2.plot(max_depths, depth_scores, label="max_depths")
    ax2.set_xlabel('Max Depth Values')
    ax2.set_ylabel('Mean Accuracy Scores')
    ax2.set_title('Random Forests - max_depth vs accuracy', fontsize=12)
    pylab.show()

    test_scores = []

    for train_percent in training_percentage:
        #     print "\nTraining Set Size : " + str(train_percent*100) + "%"
        x_train_main, x_test, y_train_main, y_test = train_test_split(
            x_data,
            y_data['class'],
            test_size=1 - train_percent,
            random_state=42
        )

        rfc_k = RandomForestClassifier(
            max_depth=max_depth_values_by_dataset[current_dataset],
            n_estimators=estimator_values_by_dataset[current_dataset],
            max_features=1)
        rfc_k.fit(x_train_main, y_train_main)
        predicted_k = rfc_k.predict(x_test)
        scores = metrics.accuracy_score(y_test, predicted_k)
        test_scores.append(scores)

    fig3 = plt.figure(figsize=(8, 6), dpi=80)
    ax3 = fig3.add_subplot(111)
    ax3.plot(training_percentage, test_scores, label="training percentage")
    ax3.set_xlabel('Training Data Fraction')
    ax3.set_ylabel('Mean Accuracy Scores')
    ax3.set_title('Random Forests - Accuracy on Test Set', fontsize=12)
    pylab.show()

    ##############################
    ####### Deep Learning ########
    ##############################
    print 'Deep Learning'
    dl.deeplearning_main(current_dataset, x_data, y_data)

    ##############################
    ######## MULTI-CLASS #########
    ######## KERNEL SVM  #########
    ##############################
    print "----Running Custom Multi-Class Kernel SVM----"
    kernel_svm_test.test(current_dataset)

if __name__ == "__main__":
    main()
