from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import datasets as ds

def deeplearning_main(current_dataset, x_data, y_data):
    #print 'Deep Learning'
    #print x_data.shape, y_data.shape
    train_percentage = [0.1, 0.2, 0.3, 0.4, 0.5]
    if current_dataset == 'breast_cancer' :
        print 'converting classes to 0-1'
        y_data = y_data.replace(2,0)
        y_data = y_data.replace(4,1)

    for train_percent in train_percentage:
        print train_percent
        x_train_main, x_test, y_train_main, y_test = train_test_split(
            x_data,
            y_data['class'],
            test_size=1 - train_percent,
            random_state=42
        )
        deeplearning_classify(current_dataset,
            x_train_main.as_matrix(), y_train_main.as_matrix(),
            x_test.as_matrix(), y_test.as_matrix())

def deeplearning_classify (dataset_name,X_Train, Y_train,X_Test, Y_test):
    if dataset_name == 'breast_cancer':
        return keras_breast_cancer_data(X_Train, Y_train,X_Test, Y_test)
    elif dataset_name == 'digits':
        return keras_digits_data(X_Train, Y_train,X_Test, Y_test)
    elif dataset_name == 'forest_mapping':
        return keras_forest_data(X_Train, Y_train,X_Test, Y_test)
    else:
        print 'no such data set'

def keras_breast_cancer_data (X_Train, Y_train,X_Test, Y_test):
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=9, init='uniform', activation='sigmoid'))
    model.add(Dense(output_dim=1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.fit(X_Train, Y_train, nb_epoch=20,verbose=0)
    loss_and_metrics = model.evaluate(X_Test, Y_test, batch_size=32)
    y_predicted = model.predict(X_Test)
    roc_auc = roc_auc_score(Y_test, y_predicted)
    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])
    print 'ROC ', roc_auc

def keras_digits_data (X_Train, Y_train,X_Test, Y_test):
    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)
    #print Y_test.shape, Y_train.shape
    model = Sequential()
    model.add(Dense(output_dim=512, input_dim=64, init='uniform', activation='sigmoid'))
    model.add(Dense(output_dim=10, activation='sigmoid'))
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.fit(X_Train, Y_train, nb_epoch=20,verbose=0)
    loss_and_metrics = model.evaluate(X_Test, Y_test, batch_size=32)
    y_predicted = model.predict(X_Test)
    #print y_predicted.shape
    roc_auc = roc_auc_score(Y_test, y_predicted)
    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])
    print 'ROC ', roc_auc

def keras_forest_data (X_Train, Y_train,X_Test, Y_test):
    Y_train = np_utils.to_categorical(Y_train, 4)
    Y_test = np_utils.to_categorical(Y_test, 4)
    model = Sequential()
    model.add(Dense(output_dim=512, input_dim=27, init='uniform', activation='sigmoid'))
    model.add(Dense(output_dim=4, activation='sigmoid'))
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.fit(X_Train, Y_train, nb_epoch=20,verbose=0)
    loss_and_metrics = model.evaluate(X_Test, Y_test, batch_size=32)
    y_predicted = model.predict(X_Test)
    roc_auc = roc_auc_score(Y_test, y_predicted)
    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])
    print 'ROC ', roc_auc

if __name__ == "__main__":
    current_dataset = 'digits'
    x_data, y_data = ds.retrieve_data_sets(current_dataset)
    deeplearning_main(current_dataset, x_data, y_data)
