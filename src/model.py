import os
import sys
from sys import exit
import numpy as np
from numpy.random import randint
from numpy import expand_dims
from numpy import delete
from numpy import zeros
from numpy import ones
from numpy import empty
from numpy import loadtxt
from numpy import asarray
from numpy import append
from numpy import array_equal
from numpy.random import randn
from numpy.random import randint
from datetime import datetime
from keras.models import Model
from keras.optimizers import Adam
import argparse
sys.path.insert(1, 'data/')
import pre_processing


# created to make sure that both discriminator models have the same wheights
def same_model(a,b):
    return any([array_equal(a1, a2) for a1, a2 in zip(a.get_weights(), b.get_weights())])

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def load_from_directory(path):
    files = os.listdir(path)
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
def load_real_labeled_samples(path):
    X_0 = load_from_directory(path+'/DF')
    X_1 = load_from_directory(path+'/SD')
    return [X_0, X_1]

# load the unlabeled data
def load_real_unlabeled_samples(path):
    return load_from_directory(path)

# Simply randomly split an array in two
# There is a bug here and the test is not 100% garanted balanced, but for now it is close enough
def split_test_data(data, test_size):
    mask = np.zeros(data.shape[0],dtype=bool)
    ix = randint(0, data.shape[0], size=test_size)
    mask[ix] = True
    X_training = data[~mask]
    X_test = data[mask]
    return X_training , X_test

# Generates Training and Test data. The test dataset is always balanced.
def generate_supervised_datasets(X, relative_test_size=0.2):
    total_size = len(X[0]) + len(X[1])
    half_test_size = int((total_size * relative_test_size) / 2 )
    X_training_0, X_test_0 = split_test_data(X[0], half_test_size)
    X_training_1, X_test_1 = split_test_data(X[1], half_test_size)
    X_training = append(X_training_0, X_training_1, axis=0)
    y_training = append(zeros((len(X_training_0), 1 )), ones((len(X_training_1), 1 )), axis=0)
    X_test = append(X_test_0, X_test_1, axis=0)
    y_test = append(zeros((len(X_test_0), 1 )), ones((len(X_test_1), 1 )), axis=0)
    return [X_training , y_training] , [X_test , y_test]

def generate_unsupervised_datasets(X, relative_test_size=0.05):
    test_size = half_test_size = int((X.shape[0] * relative_test_size))
    X_training, X_test = split_test_data(X, test_size)
    return X_training, X_test

def select_supervised_samples(dataset, n_samples=10):
    half_samples = int(n_samples/2)
    mask = np.array(dataset[1], dtype=bool)
    mask = np.reshape(mask,(dataset[0].shape[0]))
    X_0 = dataset[0][~mask]
    X_1 = dataset[0][mask]
    ix_0 = randint(0, X_0.shape[0], half_samples)
    ix_1 = randint(0, X_1.shape[0], half_samples)
    X = append(X_0[ix_0], X_1[ix_1], axis=0)
    y = append(zeros((half_samples, 1 )), ones((half_samples, 1 )), axis=0)
    return [X, y]

# select real samples
def select_unsupervised_samples(dataset, n_samples=250):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select samples and labels
    X= dataset[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, y]

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    samples = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return samples, y

def generate_unsupervised_test_dataset(g_model, latent_dim, X_real):
    y_real = ones((X_real.shape[0], 1))
    return X_real, y_real
    # X_fake, y_fake = generate_fake_samples(g_model, latent_dim, X_real.shape[0])
    # X = append(X_fake, X_real, axis=0)
    # y = append(y_fake, y_real, axis=0)
    # return X, y 

# generate samples and save as a plot and save the model
def summarize_performance(g_model, d_model, c_model, latent_dim, labeled_test_dataset, unlabeled_test_dataset, path, log, step, save_performance=False, n_samples=100):
    X, y = labeled_test_dataset
    labeled_loss, labeled_acc = c_model.evaluate(X, y, verbose=0)
    X, y = generate_unsupervised_test_dataset(g_model, latent_dim, unlabeled_test_dataset)
    unlabeled_loss, unlabeled_acc = d_model.evaluate(X, y, verbose=0)
    if(save_performance):
        # print('Classifier Accuracy: %.3f%%  |  Classifier Loss: %.3f%%' % (acc * 100, loss))
        # acc_log = 'Tests Resultsfor models in folder '+str(count)+': \nClassifier Accuracy: %.3f%%  |  Classifier Loss: %.3f%% \n\n' % (acc * 100, loss)
        new_path = path+'/'+'partial_'+str(step)+'/'
        os.mkdir(new_path)
        # save the generator model
        filename1 = new_path + 'g_model.h5'
        g_model.save(filename1)
        # save the classifier model
        filename2 = new_path + 'c_model.h5'
        c_model.save(filename2)
        filename3 = new_path + 'd_model.h5'
        d_model.save(filename3)
    return labeled_loss, labeled_acc, unlabeled_loss, unlabeled_acc

def create_mask(arr):
    mask = np.zeros(len(arr),dtype=bool)
    for i in range(len(arr)):
        if(arr[i][0] > 0.5):
            mask[i] = True
    return mask

def summarize_performance_t2(d_model, c_model, labeled_test_dataset):
    X, y = labeled_test_dataset
    n_tests = y.shape[0]
    predict_d = d_model.predict(X, verbose=0)
    mask = create_mask(predict_d)
    X = X[mask]
    y = y[mask]
    loss = 0
    acc = 0 
    unsup_acc = y.shape[0]/n_tests
    if(X.shape[0]>0):
        loss, acc = c_model.evaluate(X, y, verbose=0)
    return unsup_acc , loss, acc

def summarize_performance_t3(d_model, g_model, size = 200):
    generated, labels = generate_fake_samples(g_model, 100, size)
    predict_d = d_model.predict(generated, verbose=0)
    count = sum(predict_d>0.5)
    return count/size

# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, labeled_train_dataset, labeled_test_dataset, unlabeled_dataset, unlabeled_train_dataset, unlabeled_test_dataset, latent_dim, test_size, path, n_instance, n_epochs=1000, n_batch=100):
    # calculate the number of batches per training epoch
    bat_per_epo = int(unlabeled_train_dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    log = ''
    for i in range(n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real] = select_supervised_samples(labeled_train_dataset, n_samples=10)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, y_real] = select_unsupervised_samples(unlabeled_dataset, n_samples=half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        # log = log + '>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f] \n' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss)
        # evaluate the model performance every so often
        if (i+1) % 100 == 0:
            labeled_loss, labeled_acc, unlabeled_loss, unlabeled_acc = summarize_performance(g_model, d_model, c_model, latent_dim, labeled_test_dataset, unlabeled_test_dataset, path, log, i+1, save_performance=True)
            t2_measured, t2_loss, t2_acc = summarize_performance_t2(d_model, c_model, labeled_test_dataset)
            log = log + str(n_instance+1)+','+str(i+1)+','+str(labeled_loss)+','+str(labeled_acc)+','+str(unlabeled_loss)+','+str(unlabeled_acc) +','+str(t2_measured) +','+str(t2_loss) +','+str(t2_acc)
            t3_acc = summarize_performance_t3(d_model, g_model)
            log = log + ' | ' + str(t3_acc)+'\n'
            print(str(i+1), labeled_loss, labeled_acc, unlabeled_loss, unlabeled_acc,' | ',t2_measured, t2_loss, t2_acc, ' | ', t3_acc)
    return log

def train_instances(labeled_dataset, unlabeled_dataset, net_model, path, n_instances = 1):
    model_name = str(net_model).split()[1]
    # path to save logs, performances and fake samples files
    path = path + 'run/'
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + 'test_'+model_name+'_'+datetime.now().isoformat()
    os.mkdir(path)
    print("[INFO] Labeled Dataset size: ", len(labeled_dataset[0])+len(labeled_dataset[1])) 
    print("[INFO] Unlabeled Dataset size: ", len(unlabeled_dataset)) 
    # log summary
    log = ''
    log = log + 'instance,step,labeled_loss,labeled_acc,unlabeled_loss,unlabeled_acc\n'
    for i in range(n_instances):
        # size of the latent space
        latent_dim = 100
        # create the discriminator models
        d_model, c_model = net_model.define_discriminator()
        # create the generator
        g_model = net_model.define_generator(latent_dim)
        # create the gan
        gan_model = define_gan(g_model, d_model)
        # generate train and test LABELED datasets
        labeled_train_dataset, labeled_test_dataset = generate_supervised_datasets(labeled_dataset)
        # generate train and test UNLABELED datasets
        unlabeled_train_dataset, unlabeled_test_dataset = generate_unsupervised_datasets(unlabeled_dataset)
        # relative size of the test data
        test_size = 0.2
        # train model
        train_log = train(g_model, d_model, c_model, gan_model, labeled_train_dataset, labeled_test_dataset, unlabeled_dataset, unlabeled_train_dataset, unlabeled_test_dataset, latent_dim, test_size, path, i)
        # uptade the log
        log = log + train_log
    log_name = path +'.log'
    log_file = open(log_name, "w")
    log_file.write(log)
    log_file.close()