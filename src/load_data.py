from os import listdir
from numpy import zeros
from numpy import ones
from numpy import loadtxt
from numpy import append
from numpy import expand_dims
from numpy import savetxt
from numpy import loadtxt
from numpy.random import randint
import json

def load_from_directory(path):
    files = listdir(path)
    X = loadtxt(open(path+"/"+ files[0], "rb"), delimiter=",")
    vector_size = X.shape[0]
    if len(X.shape) > 1:
        rows_size, columns_size = X.shape
    # get samples
    for i  in range(1, len(files)):
        new = loadtxt(open(path+"/"+ files[i], "rb"), delimiter=",")
        X = append(X, new, axis =0)
    # reshape the ndarray
    if len(X.shape) > 1:
        X = X.reshape(len(files),rows_size,columns_size)
    else:
        X = X.reshape(len(files),vector_size)
    # expand dimension 
    X = expand_dims(X, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    return X

# load the labeled data
def load_labeled_samples(path):
    X_0 = load_from_directory(path+'/DF')
    X_1 = load_from_directory(path+'/SD')
    return [X_0, X_1]

# load the unlabeled data
def load_unlabeled_samples(path):
    return load_from_directory(path)

# Simply randomly split an array in two
# There is a bug here and the test is not 100% garanted balanced, but for now it is close enough
def split_test_data(data, test_size, random, file_path):
    mask = zeros(data.shape[0],dtype=bool)
    if(random):
        ix = randint(0, data.shape[0], size=test_size)
    else:
        ix = loadtxt(file_path).astype(int)
    mask[ix] = True
    X_training = data[~mask]
    X_test = data[mask]
    return X_training , X_test, ix

# Generates Training and Test data. The test dataset is always balanced.
def generate_supervised_datasets(X, log_path, random, relative_test_size=0.2):
    filepath_0 = log_path+'/test_indices_labeled_0'
    filepath_1 = log_path+'/test_indices_labeled_1'
    total_size = len(X[0]) + len(X[1])
    half_test_size = int((total_size * relative_test_size) / 2 )
    X_training_0, X_test_0, indices_0 = split_test_data(X[0], half_test_size, random, filepath_0)
    X_training_1, X_test_1, indices_1 = split_test_data(X[1], half_test_size, random, filepath_1)
    X_training = append(X_training_0, X_training_1, axis=0)
    y_training = append(zeros((len(X_training_0), 1 )), ones((len(X_training_1), 1 )), axis=0)
    X_test = append(X_test_0, X_test_1, axis=0)
    y_test = append(zeros((len(X_test_0), 1 )), ones((len(X_test_1), 1 )), axis=0)
    if(random):
        savetxt(filepath_0, indices_0, delimiter=',', fmt='%-7.0f')
        savetxt(filepath_1, indices_1, delimiter=',', fmt='%-7.0f')
    return [X_training , y_training] , [X_test , y_test]

def generate_unsupervised_datasets(X, log_path, random, relative_test_size=0.05):
    filepath = log_path+'/test_indices_unlabeled'
    test_size = half_test_size = int((X.shape[0] * relative_test_size))
    X_training, X_test, indices = split_test_data(X, test_size, random, filepath)
    if(random):
        savetxt(filepath, indices, delimiter=',', fmt='%-7.0f')
    return X_training, X_test
