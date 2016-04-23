from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

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
    model.fit(X_Train, Y_train, nb_epoch=20,verbose=1)
    loss_and_metrics = model.evaluate(X_Test, Y_test, batch_size=32)
    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])

def keras_digits_data (X_Train, Y_train,X_Test, Y_test):
    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)
    model = Sequential()
    model.add(Dense(output_dim=512, input_dim=64, init='uniform', activation='sigmoid'))
    model.add(Dense(output_dim=10, activation='sigmoid'))
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.fit(X_Train, Y_train, nb_epoch=20,verbose=1)
    loss_and_metrics = model.evaluate(X_Test, Y_test, batch_size=32)
    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])

def keras_forest_data (X_Train, Y_train,X_Test, Y_test):
    Y_train = np_utils.to_categorical(Y_train, 4)
    Y_test = np_utils.to_categorical(Y_test, 4)
    model = Sequential()
    model.add(Dense(output_dim=512, input_dim=27, init='uniform', activation='sigmoid'))
    model.add(Dense(output_dim=4, activation='sigmoid'))
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.fit(X_Train, Y_train, nb_epoch=20,verbose=1)
    loss_and_metrics = model.evaluate(X_Test, Y_test, batch_size=32)
    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])
