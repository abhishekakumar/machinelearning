
import deeplearning as dl
import kernel_svm_test
import knn
import kmeans
import svm
import randomforests as rf
import datasets as ds
import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")

training_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    datasets = ['breast_cancer', 'digits', 'forest_mapping']
    for current_dataset in datasets:
        classify(current_dataset);
    # classify('digits')


def classify(current_dataset):
    print current_dataset
    x_data, y_data = ds.retrieve_data_sets(current_dataset)

    # deep learning uses non-normalized data for forest mapping
    if current_dataset == 'forest_mapping':
        x_data_dl, y_data_dl = ds.retrieve_forest_data_dl()

    ##############################
    ### Support Vector Machine ###
    ##############################
    print 'SVM'
    svm_accuracies = svm.SVM(current_dataset)

    #############################
    ###### Random Forests #######
    #############################
    print 'Random Forests'
    rf_accuracies = rf.randomforests(current_dataset)

    # ##############################
    # ####### Deep Learning ########
    # ##############################
    print 'Deep Learning'
    if current_dataset == 'forest_mapping':
        dl_accuracies = dl.deeplearning_main(current_dataset, x_data_dl, y_data_dl)
    else:
        dl_accuracies = dl.deeplearning_main(current_dataset, x_data, y_data)

    # ##############################
    # ######## MULTI-CLASS #########
    # ######## KERNEL SVM  #########
    # ##############################
    print "----Running Custom Multi-Class Kernel SVM----"
    kernel_svm_accuracies = kernel_svm_test.test(current_dataset)

    # ##############################
    # #### K Nearest Neighbors #####
    # ##############################
    print 'K Nearest Neighbors'
    knn_accuracies = knn.test_knn(current_dataset)

    # ##############################
    # #### K Nearest Neighbors #####
    # ##############################
    print 'K-Means'
    kmeans_accuracies = kmeans.test_kmeans(current_dataset)

    # ##############################
    # #### Graph Plot #####
    # ##############################

    print '\nPlotting Graph. Please close the graph to continue'
    red_patch = mpatches.Patch(color='red', label='sklearn SVM')
    yellow_patch = mpatches.Patch(color='yellow', label='Deep Learning')
    cyan_patch = mpatches.Patch(color='cyan', label='Random Forests')
    black_patch = mpatches.Patch(color='black', label='kernel SVM')
    blue_patch = mpatches.Patch(color='blue', label='kNN')
    green_patch = mpatches.Patch(color='green', label='K Means')

    plt.figure(figsize=(10, 8))
    plt.plot(training_percentage, svm_accuracies, 'r', linewidth=2.0)
    plt.plot(training_percentage, dl_accuracies, 'y', linewidth=2.0)
    plt.plot(training_percentage, rf_accuracies, 'c', linewidth=2.0)
    plt.plot(training_percentage, kernel_svm_accuracies, 'k', linewidth=2.0)
    plt.plot(training_percentage, knn_accuracies, 'b', linewidth=2.0)
    plt.plot(training_percentage, kmeans_accuracies, 'g', linewidth=2.0)
    plt.xlabel('training set fraction')
    plt.ylabel('test set accuracy')
    plt.title('test set accuracy on data set : '+current_dataset)
    plt.legend(handles=[red_patch, yellow_patch, cyan_patch, black_patch, blue_patch, green_patch], loc=4)
    pylab.show()


if __name__ == "__main__":
    main()
