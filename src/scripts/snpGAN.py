import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import genfromtxt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import sys


##          ============================================
##                          IMPORTING DATA
##          ============================================

##  Visualize a sample as a image
# sample = genfromtxt('../data/real/sample_1.csv', delimiter=',')

# fig = plt.figure(figsize = (3,3)) 
# img = fig.add_subplot(111)
# img.imshow(sample, cmap='viridis')
# plt.savefig('../data/images/real_sample_1.png')

##  Obtain the dataset
original_samples = []
for i in range(1,201):
    sample = genfromtxt('../data/real/sample_'+str(i)+'.csv', delimiter=',')
    original_samples.append(sample)

##  Generate the training datasets
def train_loader(samples, batch_size):
    # return np.split(np.array(samples), batch_size)
    batches = []
    batch = []
    count = 1
    for i in range(0,200):
        batch.append([samples[i]])
        if count == batch_size:
            batches.append(batch)
            count = 0
            batch = []
        count += 1
    return torch.from_numpy(np.array(batches))

##          ============================================
##                      CREATING MODELS
##          ============================================

##  Define the Discriminator
class Discriminator(nn.Module):
    
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        
        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        
        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim, output_size)
        
        # dropout layer 
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        # flatten sample
        x = x.view(-1, 10*10).float()
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer
        out = self.fc4(x)

        return out
    
##  Define the Generator
class Generator(nn.Module):
    
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        
        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        
        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
        
        # dropout layer 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer with tanh applied
        out = F.tanh(self.fc4(x))

        return out
    
##  Discriminator hyperparams

##  Size of input sample to discriminator (10*10)
input_size = 100
##  Size of discriminator output (real or fake)
d_output_size = 1
##  Size of last hidden layer in the discriminator
d_hidden_size = 10

##  Generator hyperparams

##  Size of latent vector to give to generator
z_size = 100
##  Size of discriminator output (generated sample)
g_output_size = 100
##  Size of first hidden layer in the generator
g_hidden_size = 10

##  Instantiate discriminator and generator
D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

##  Calculate losses
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
##  Label smoothing
    if smooth:
##      Smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
##      Real labels = 1
        labels = torch.ones(batch_size)
        
##  Numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
##  Calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
##  Fake labels = 0
    labels = torch.zeros(batch_size)
    criterion = nn.BCEWithLogitsLoss()
##  Calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


##  We want to update the generator and discriminator variables separately:
##  Optimizers
lr = 0.002

##  Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr)


##          ============================================
##                          TRAINING
##          ============================================

##  Training hyperparams
num_epochs = 100

##  Keep track of loss and generated, "fake" samples
samples = []
losses = []

##  Get some fixed data for sampling. These are samples that are held
##  constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

##  Size of each traning dataset
batch_size = 20
num_batches = len(original_samples)/batch_size

##  Train the network
D.train()
G.train()
for epoch in range(num_epochs):
    batch_i = 1
    for training_samples in train_loader(original_samples, batch_size):
        
##      Rescale input samples from [0,1) to [-1, 1)
        training_samples = training_samples*2 - 1
        
##      ============================================
##                 TRAIN THE DISCRIMINATOR
##      ============================================
        
        d_optimizer.zero_grad()
        
        # 1. Train with real samples

        # Compute the discriminator losses on real samples 
        # smooth the real labels
        D_real = D(training_samples)
        d_real_loss = real_loss(D_real, smooth=True)
        
        # 2. Train with fake samples
        
        # Generate fake samples
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_samples = G(z)
        
        # Compute the discriminator losses on fake samples        
        D_fake = D(fake_samples)
        d_fake_loss = fake_loss(D_fake)
        
        # add up loss and perform backprop
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        
##      =========================================
##                 TRAIN THE GENERATOR
##      =========================================
        g_optimizer.zero_grad()
        
        # 1. Train with fake samples and flipped labels
        
        # Generate fake samples
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_samples = G(z)
        
        # Compute the discriminator losses on fake samples 
        # using flipped labels!
        D_fake = D(fake_samples)
        g_loss = real_loss(D_fake) # use real loss to flip labels
        
        # perform backprop
        g_loss.backward()
        g_optimizer.step()

        if batch_i == num_batches:
        # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))
            batch_i = 0
        batch_i += 1

    
    ## AFTER EACH EPOCH##
    # append discriminator loss and generator loss
    losses.append((d_loss.item(), g_loss.item()))
    
    # generate and save sample, fake samples
    G.eval() # eval mode for generating samples
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train() # back to train mode

##  Plot the losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.savefig('../data/images/training_losses.png')

##  Save training generator samples
# with open('train_samples.pkl', 'wb') as f:
#     pkl.dump(samples, f)

##  View some fake samples
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((10,10)), cmap='viridis')
        plt.savefig('../data/images/fake_samples.png')
        
view_samples(-1, samples)