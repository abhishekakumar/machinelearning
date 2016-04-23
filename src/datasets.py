import pandas as pd
import os
import numeric_class as nc


def retrieve_data_sets(dataset_name):
    if dataset_name == 'breast_cancer':
        return get_breast_cancer_data()
    elif dataset_name == 'digits':
        return get_digits_data()
    elif dataset_name == 'forest_mapping':
        return get_forest_data()
    else:
        print 'no such data set'


def get_breast_cancer_data():
    subfolder = 'train_data/breastcancer'
    breast_cancer_data = pd.read_csv(os.path.join(subfolder, 'breast-cancer-wisconsin.data'), sep=",", header=None)
    breast_cancer_data.columns = ['id', 'thickness', 'cell_size', 'cell_shape', 'adhesion',
                                  'single_cell_size', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'class']
    x_columns = ['thickness', 'cell_size', 'cell_shape', 'adhesion',
                 'single_cell_size', 'nuclei', 'chromatin', 'nucleoli', 'mitoses']

    y_columns = ['class']
    wrong_data = breast_cancer_data.nuclei == '?'
    right_data = breast_cancer_data.nuclei != '?'

    nuclei_mean = breast_cancer_data.nuclei[right_data].astype(int).mean()
    breast_cancer_data.nuclei[wrong_data] = nuclei_mean
    return breast_cancer_data[x_columns], breast_cancer_data[y_columns]


def get_digits_data():
    subfolder = 'train_data/digits'
    X_temp = []
    y_temp = []
    X = []
    y = []

    with open(os.path.join(subfolder, 'optdigits-orig.tra'), 'r') as text_file:
        for line in text_file:
            if line.startswith('0') or line.startswith('1'):
                line = line[:-1]  # remove \n here
                X_temp.extend(list(line))
            elif line.startswith(' '):
                line = line[1:-1]
                y_temp.append(line)
                X.append(X_temp)
                y.append(y_temp)
                X_temp = []
                y_temp = []

    x_data = pd.DataFrame(X)
    y_data = pd.DataFrame(y)
    y_data.columns = ['class']
    return x_data, y_data


def get_forest_data():
    subfolder = 'train_data/forest_mapping'
    forest_data = pd.read_csv(os.path.join(subfolder, 'training.csv'), sep=",", header=0)

    forest_data['numeric_class'] = forest_data.apply(lambda row: nc.get_numeric_class(row), axis=1)
    x_data = forest_data.ix[:, 1:28]
    y_data = forest_data.ix[:, 28]
    y_data = pd.DataFrame(y_data)
    y_data.columns = ['class']

    #x_data_norm = (x_data - x_data.mean()) / (x_data.max() - x_data.min())
    return x_data, y_data
