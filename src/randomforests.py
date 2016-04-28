from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

import datasets as ds
import plotgraph as pg

k_fold_values = {'breast_cancer': 10, 'digits': 10, 'forest_mapping': 6}

# Set the values below from Cross Validation Graphs
estimator_values_by_dataset = {'breast_cancer': 100, 'digits': 1000, 'forest_mapping': 50}
max_depth_values_by_dataset = {'breast_cancer': 20, 'digits': 50, 'forest_mapping': 100}

estimators = [10, 20, 50, 100, 500, 1000]
max_depths = [5, 10, 20, 50, 100, 200]

train_percentage = [0.1, 0.2, 0.3, 0.4, 0.5]
training_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # For plotting the training accuracy


def randomforests(current_dataset):

    x_data, y_data = ds.retrieve_data_sets(current_dataset)

    for train_percent in train_percentage:
        print "Training on percent: ", train_percent
        depth_scores = []
        estimator_scores = []
        x_train_main, x_test, y_train_main, y_test = train_test_split(
            x_data,
            y_data['class'],
            test_size=1 - train_percent,
            random_state=42
        )

        for max_depth in max_depths:
            rfc_k = RandomForestClassifier(max_depth=max_depth, n_estimators=500, max_features=1)
            scores = cross_val_score(rfc_k, x_train_main, y_train_main, cv=k_fold_values[current_dataset],
                                     scoring='accuracy')
            depth_scores.append(scores.mean())

        for estimator in estimators:
            rfc_k = RandomForestClassifier(max_depth=100, n_estimators=estimator, max_features=1)
            scores = cross_val_score(rfc_k, x_train_main, y_train_main, cv=k_fold_values[current_dataset],
                                     scoring='accuracy')
            estimator_scores.append(scores.mean())

    # pg.plot(
    #     estimators,
    #     estimator_scores,
    #     'Estimator Values',
    #     'Mean Accuracy Scores',
    #     'Random Forests - estimator vs accuracy - Dataset : ' + current_dataset,
    #     'linear'
    # )
    #
    # pg.plot(
    #     max_depths,
    #     depth_scores,
    #     'Max Depth Values',
    #     'Mean Accuracy Scores',
    #     'Random Forests - max_depth vs accuracy - Dataset : ' + current_dataset,
    #     'linear'
    # )

    test_scores = []

    for train_percent in training_percentage:
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

    # print 'RF : Data Set : ' + current_dataset + " Accuracy : " + str(test_scores[4])
    # pg.plot(
    #     training_percentage,
    #     test_scores,
    #     'Training Data Fraction',
    #     'Accuracy - Test Set',
    #     'Random Forests - Accuracy on Test Set - Dataset : ' + current_dataset,
    #     'linear'
    # )
    return test_scores

