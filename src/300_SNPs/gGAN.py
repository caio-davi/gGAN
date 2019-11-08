# example of semi-supervised gan for mnist
import os
import sys
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

# for analysis of the tests of the supervised discriminator
def accuracy_analysis(y, arr):
    a = []
    places = []
    count = 0
    for x in arr:
        if(x[0] > x[1]):
            a.append(0)
        else:
            a.append(1)
    for i in range(0, len(y)):
        if(y[i] == a[i]):
            count += 1
            places.append(i)
    print( 'correct: ', count)
    print( 'acc: ', count/len(y))
    print( 'places: ', places)
    return a;

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
    c_out_layer = Activation('sigmoid')(fe)
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
    cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    files = os.listdir(path)
    X = loadtxt(open(path+"/"+ files[0], "rb"), delimiter=",", skiprows=1, usecols=cols)
    # get samples
    for i  in range(1, len(files)):
        new = loadtxt(open(path+"/"+ files[i], "rb"), delimiter=",", skiprows=1, usecols=cols)
        X = append(X, new, axis =0)
    # reshape the ndarray
    X = X.reshape(len(files),12,20)
    # expand dimension 
    X = expand_dims(X, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    return X

# load the labeled data
def load_real_labeled_samples():
    X_training_0 = load_from_directory('./data/labeled/training/DF')
    X_training_1 = load_from_directory('./data/labeled/training/SD')
    X_test_0 = load_from_directory('./data/labeled/test/DF')
    X_test_1 = load_from_directory('./data/labeled/test/SD')
    return [X_training_0, X_training_1, X_test_0, X_test_1]

# load the unlabeled data
def load_real_unlabeled_samples():
    return load_from_directory('./data/unlabeled')

# load the unlabeled data
# def load_real_unlabeled_samples(test=False):
#     path = './data/unlabeled'
#     cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#     len_unlabeled = len(os.listdir(path))
#     X = loadtxt(open(path+"/sample_0.csv", "rb"), delimiter=",", skiprows=1, usecols=cols)
#     # get samples
#     for i in range(1, len_unlabeled):
#         new = loadtxt(open(path+"/sample_"+str(i)+".csv", "rb"), delimiter=",", skiprows=1, usecols=cols)
#         X = append(X, new, axis =0)
#     # reshape the ndarray
#     X = X.reshape(len_unlabeled,12,20)
#     # expand dimension 
#     X = expand_dims(X, axis=-1)
#     # convert from ints to floats
#     X = X.astype('float32')
#     return X

# select a supervised subset of the dataset
def select_supervised_samples(dataset, test_data=False, n_samples=10):
    if(test_data):
        X_0 = dataset[2]
        X_1 = dataset[3]
        y_0 = zeros((len(X_0), 1 ))
        y_1 = ones((len(X_1), 1 ))
    else:
        X_training_0 = dataset[0]
        X_training_1 = dataset[1]
        half_samples = int(n_samples/2)
        ix = randint(0, X_training_0.shape[0], size=half_samples)
        X_0= X_training_0[ix]
        ix = randint(0, X_training_1.shape[0], size=half_samples)
        X_1= X_training_1[ix]
        y_0 = zeros((half_samples, 1 ))
        y_1 = ones((half_samples, 1 ))
    X = append(X_0, X_1, axis=0)
    y = append(y_0, y_1, axis=0)
    return [X, y]

# select real samples
def select_unsupervised_samples(dataset, n_samples=250):
    # split into images and labels
    samples = dataset
    # choose random instances
    ix = randint(0, samples.shape[0], n_samples)
    # select images and labels
    X= samples[ix]
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
def summarize_performance(step, g_model, c_model, latent_dim, dataset, path, log, count, n_samples=100):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='viridis')
    # save plot to file
    new_path = path+'/'+str(count)+'/'
    os.mkdir(new_path)
    filename1 = new_path + '/generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # evaluate the classifier model
    X, y = select_supervised_samples(dataset, test_data=True)
    # _, acc = c_model.evaluate(X, y, verbose=0)
    loss, acc = c_model.test_on_batch(X, y)
    print('Classifier Accuracy: %.3f%%  |  Classifier Loss: %.3f%%' % (acc * 100, loss))
    log = log + '\n' + 'Classifier Accuracy: %.3f%%' % (acc * 100)
    filename4 = new_path + 'logs'
    log_file = open(filename4, "w")
    log_file.write(log)
    log_file.close()
    # save the generator model
    filename2 = new_path + 'g_model_%04d.h5' % (step+1)
    g_model.save(filename2)
    # save the classifier model
    filename3 = new_path + 'c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, labeled_dataset, unlabeled_dataset, latent_dim, n_epochs=10, n_batch=100):
    # log summary
    log = ''
    # path to save logs, performances and fake samples files
    path = './run/'+datetime.now().isoformat()+'/'
    os.mkdir(path)
    # calculate the number of batches per training epoch
    bat_per_epo = int(unlabeled_dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d \n' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    full_log = 'n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d \n' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps)
    full_log = full_log + '\n\n'
    count = 0
    # manually enumerate epochs
    for i in range(n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real] = select_supervised_samples(labeled_dataset)
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
        log = log + '>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f] \n' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss)
        full_log = full_log + '>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f] \n' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss)
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, c_model, latent_dim, labeled_dataset, path, log, count)
            count += 1
            log = ''
    filename = path + 'full_training_log'
    log_file = open(filename, "w")
    log_file.write(full_log)
    log_file.close()
 
# size of the latent space
latent_dim = 100
# create the discriminator models
d_model, c_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load labeled data
labeled_dataset = load_real_labeled_samples()
# load unlabeled data
unlabeled_dataset = load_real_unlabeled_samples()
# set pointer to the labeled data
next_supervised = 0
# train model
train(g_model, d_model, c_model, gan_model, labeled_dataset, unlabeled_dataset, latent_dim)