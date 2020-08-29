from os import mkdir
import numpy as np
from numpy import delete
from numpy import zeros
from numpy import ones
from numpy import empty
from numpy import loadtxt
from numpy import asarray
from numpy import append
from numpy import array
from numpy import reshape
from numpy.random import randn
from numpy.random import randint
from keras.models import Model
from keras.optimizers import Adam

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

def select_supervised_samples(dataset, n_samples=10):
    half_samples = int(n_samples/2)
    mask = array(dataset[1], dtype=bool)
    mask = reshape(mask,(dataset[0].shape[0]))
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
def summarize_performance(g_model, d_model, c_model, latent_dim, labeled_test_dataset, unlabeled_test_dataset, path, step, save_performance=False, n_samples=100):
    X, y = labeled_test_dataset
    labeled_loss, labeled_acc = c_model.evaluate(X, y, verbose=0)
    X, y = generate_unsupervised_test_dataset(g_model, latent_dim, unlabeled_test_dataset)
    unlabeled_loss, unlabeled_acc = d_model.evaluate(X, y, verbose=0)
    if(save_performance):
        # print('Classifier Accuracy: %.3f%%  |  Classifier Loss: %.3f%%' % (acc * 100, loss))
        # acc_log = 'Tests Resultsfor models in folder '+str(count)+': \nClassifier Accuracy: %.3f%%  |  Classifier Loss: %.3f%% \n\n' % (acc * 100, loss)
        new_path = path+'/'+'partial_'+str(step)+'/'
        mkdir(new_path)
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
    mask = zeros(len(arr),dtype=bool)
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

def write_on_log(path, text, end_section = False):
    with open(path, "a") as file_object:
        file_object.write(text)
        if(end_section):
            file_object.write("---")
        file_object.write("\n")

def write_model_on_log(path, model, end_section = False):
    with open(path, "a") as file_object:
        model.summary(print_fn=lambda x: file_object.write(x + '\n'))
        if(end_section):
            file_object.write("---")
        file_object.write("\n")

def tests(g_model, d_model, c_model, latent_dim, labeled_test_dataset, unlabeled_test_dataset, path=None, step=0, save_performance=False, n_instance=0, logging=False ):
    labeled_loss, labeled_acc, unlabeled_loss, unlabeled_acc = summarize_performance(g_model, d_model, c_model, latent_dim, labeled_test_dataset, unlabeled_test_dataset, path, step, save_performance)
    t2_measured, t2_loss, t2_acc = summarize_performance_t2(d_model, c_model, labeled_test_dataset)
    t3_acc = summarize_performance_t3(d_model, g_model)
    if(logging):
        log_path = path +'.md'
        write_on_log(log_path, "| "+str(n_instance+1)+' | '+str(step)+' | '+str(labeled_loss)+' | '+str(labeled_acc)+' | '+str(unlabeled_loss)+' | '+str(unlabeled_acc) +' | '+str(t2_measured) +' | '+str(t2_loss) +' | '+str(t2_acc) + ' | ' + str(t3_acc))
    else:
        print('Test 01 - Labeled Acc: ',labeled_acc)
        print('Test 01 - Labeled Loss: ',labeled_loss)
        print('Test 01 - Unabeled Acc: ',unlabeled_acc)
        print('Test 01 - Unabeled Loss: ',unlabeled_loss)
        print('Test 02 - Recognized: ',t2_measured)
        print('Test 02 - Acc: ',t2_acc)
        print('Test 03 - Loss: ',t2_loss)
        return None

# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, datasets, latent_dim, test_size, path, n_instance, n_epochs=10000, n_batch=100):
    # calculate the number of batches per training epoch
    bat_per_epo = int(datasets['unlabeled_train_dataset'].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    for i in range(1, n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real] = select_supervised_samples(datasets['labeled_train_dataset'], n_samples=10)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, y_real] = select_unsupervised_samples(datasets['unlabeled_dataset'], n_samples=half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        # log = log + '>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f] \n' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss)
        # evaluate the model performance every so often
        if (i) % 100 == 0:
            tests(g_model, d_model, c_model, latent_dim, datasets['labeled_test_dataset'], datasets['unlabeled_test_dataset'], path, i, save_performance=True, n_instance=n_instance, logging=True)


def train_instances(datasets, net_model, path, n_instances = 1):
    latent_dim = 100
    d_model, c_model = net_model.define_discriminator()
    g_model = net_model.define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    log_path = path +'.md'
    write_on_log(log_path, "# Models Summary \n\n")
    write_on_log(log_path, "## GAN Model \n")
    write_model_on_log(log_path, c_model, True)
    write_on_log(log_path, "## Supervised Discriminator \n")
    write_model_on_log(log_path, c_model, True)
    write_on_log(log_path, "## Unsupervised Discriminator \n")
    write_model_on_log(log_path, d_model, True)
    write_on_log(log_path, "## Generator \n")
    write_model_on_log(log_path, g_model, True)
    write_on_log(log_path, "Labeled Dataset size (train): "  +str(len(datasets['labeled_train_dataset'][1]))) 
    write_on_log(log_path, "Labeled Dataset size (test): "   +str(len(datasets['labeled_test_dataset'][1]))) 
    write_on_log(log_path, "Unlabeled Dataset size (train): "+str(len(datasets['unlabeled_train_dataset']))) 
    write_on_log(log_path, "Unlabeled Dataset size: (test)"  +str(len(datasets['unlabeled_test_dataset'])), True) 
    write_on_log(log_path, "## Tests \n")
    write_on_log(log_path, '|instance|step|labeled_loss|labeled_acc|unlabeled_loss|unlabeled_acc|T2 unlabeled_acc|T2 loss|T2 labeled_acc|T3 acc|') 
    write_on_log(log_path, '|--------|----|------------|-----------|--------------|-------------|----------------|-------|--------------|------|') 
    for i in range(n_instances):
        d_model, c_model = net_model.define_discriminator()
        g_model = net_model.define_generator(latent_dim)
        gan_model = define_gan(g_model, d_model)
        test_size = 0.2
        train_log = train(g_model, d_model, c_model, gan_model, datasets, latent_dim, test_size, path, i)
    