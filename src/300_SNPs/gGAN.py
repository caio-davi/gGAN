# example of semi-supervised gan for mnist
import os
import sys
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
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from keras.utils.vis_utils import plot_model
from keras import backend
from matplotlib import pyplot
from datetime import datetime

# custom activation function
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

# created to make sure the discriminated models had the same wheights
def same_model(a,b):
    return any([array_equal(a1, a2) for a1, a2 in zip(a.get_weights(), b.get_weights())])

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(12,20,1), n_classes=2):
    # image input
    in_sample = Input(shape=in_shape)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_sample)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output layer nodes
    fe = Dense(n_classes)(fe)
    # supervised output
    c_out_layer = Activation('softmax')(fe)
    # define and compile supervised discriminator model
    c_model = Model(in_sample, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    # unsupervised output
    d_out_layer = Lambda(custom_activation)(fe)
    # define and compile unsupervised discriminator model
    d_model = Model(in_sample, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return d_model, c_model

##### plot the Discriminator
# d_model, c_model = define_discriminator()
# plot_model(c_model, to_file='discriminator1_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(d_model, to_file='discriminator2_plot.png', show_shapes=True, show_layer_names=True)

# define the standalone generator model
def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 128 * 3 * 5
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((3, 5, 128))(gen)
    # upsample to 14x14
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    # define model
    model = Model(in_lat, out_layer)
    return model

##### plot the Generator
# g_model = define_generator(100)
# plot_model(g_model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
# sys.exit()

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def load_from_directory(path):
    files = os.listdir(path)
    X = loadtxt(open(path+"/"+ files[0], "rb"), delimiter=",", skiprows=1)
    # get samples
    for i  in range(1, len(files)):
        new = loadtxt(open(path+"/"+ files[i], "rb"), delimiter=",", skiprows=1)
        X = append(X, new, axis =0)
    # reshape the ndarray
    X = X.reshape(len(files),12,20)
    # expand dimension 
    X = expand_dims(X, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    return X

# load the labeled data
# def load_real_labeled_samples():
#     X_training_0 = load_from_directory('./data/labeled/training/DF')
#     X_training_1 = load_from_directory('./data/labeled/training/SD')
#     X_test_0 = load_from_directory('./data/labeled/test/DF')
#     X_test_1 = load_from_directory('./data/labeled/test/SD')
#     return [X_training_0, X_training_1, X_test_0, X_test_1]

# load the labeled data
def load_real_labeled_samples():
    X_0 = load_from_directory('./data/labeled/DF')
    X_1 = load_from_directory('./data/labeled/SD')
    return [X_0, X_1]

# load the unlabeled data
def load_real_unlabeled_samples():
    return load_from_directory('./data/unlabeled')

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

def select_supervised_samples(dataset, n_samples=10):
    half_samples = int(n_samples/2)
    mask = np.array(dataset[1], dtype=bool)
    mask = np.reshape(mask,(dataset[0].shape[0]))
    X_0 = dataset[0][mask]
    X_1 = dataset[0][~mask]
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
    images = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, latent_dim, test_dataset, path, log, count, save_performance=False, n_samples=100):
    X, y = test_dataset
    loss, acc = c_model.evaluate(X, y, verbose=0)
    if(save_performance):
        print('Classifier Accuracy: %.3f%%  |  Classifier Loss: %.3f%%' % (acc * 100, loss))
        # acc_log = 'Tests Resultsfor models in folder '+str(count)+': \nClassifier Accuracy: %.3f%%  |  Classifier Loss: %.3f%% \n\n' % (acc * 100, loss)
        new_path = path+'/'+'partial_'+str(count)+'/'
        os.mkdir(new_path)
        # save the generator model
        filename1 = new_path + 'g_model_%04d.h5' % (step+1)
        g_model.save(filename1)
        # save the classifier model
        filename2 = new_path + 'c_model_%04d.h5' % (step+1)
        c_model.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))
    return loss, acc


# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, labeled_train_dataset, labeled_test_dataset, unlabeled_dataset, latent_dim, test_size, path, n_instance, n_epochs=10, n_batch=200):
    # calculate the number of batches per training epoch
    bat_per_epo = int(unlabeled_dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    log = ''
    for i in range(n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real] = select_supervised_samples(labeled_train_dataset)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, y_real] = select_unsupervised_samples(unlabeled_dataset)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        # log = log + '>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f] \n' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss)
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            loss, acc = summarize_performance(i, g_model, c_model, latent_dim, labeled_test_dataset, path, log, i+1)
            log = log + str(n_instance+1)+','+str(i+1)+','+str(loss)+','+str(acc)+'\n'
    return log

def batch_train(labeled_dataset, unlabeled_dataset, n_models = 10):
    # path to save logs, performances and fake samples files
    path = './run/'
    # log summary
    log = ''
    log = log + 'instance,step,loss,acc\n'
    for i in range(n_models):
        # size of the latent space
        latent_dim = 100
        # create the discriminator models
        d_model, c_model = define_discriminator()
        # create the generator
        g_model = define_generator(latent_dim)
        # create the gan
        gan_model = define_gan(g_model, d_model)
        # generate train and test datasets
        labeled_train_dataset, labeled_test_dataset = generate_supervised_datasets(labeled_dataset)
        # relative size of the test data
        test_size = 0.2
        # train model
        train_log = train(g_model, d_model, c_model, gan_model, labeled_train_dataset, labeled_test_dataset, unlabeled_dataset, latent_dim, test_size, path, i)
        # uptade the log
        log = log + train_log
    log_name = path + 'test_'+datetime.now().isoformat()+'.log'
    log_file = open(log_name, "w")
    log_file.write(log)
    log_file.close()


# load  data
labeled_dataset = load_real_labeled_samples()
unlabeled_dataset = load_real_unlabeled_samples()

batch_train(labeled_dataset,unlabeled_dataset)