from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import UpSampling1D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils.vis_utils import plot_model

# created to make sure that both discriminator models have the same wheights
def same_model(a,b):
    return any([array_equal(a1, a2) for a1, a2 in zip(a.get_weights(), b.get_weights())])

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(25,1), n_classes=2):
    # image input
    in_sample = Input(shape=in_shape)
    # fully connected layers
    fe = Dense(50, activation='relu')(in_sample)
    fe = Dense(100, activation='relu')(fe)
    fe = Dense(50, activation='relu')(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.2)(fe)
    # output layer nodes
    fe = Dense(n_classes)(fe)
    # supervised output
    c_out_layer = Activation('softmax')(fe)
    # define and compile supervised discriminator model
    c_model = Model(in_sample, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    # unsupervised output
    # d_out_layer = Lambda(custom_activation)(fe)
    d_out_layer =  Dense(1, activation='sigmoid')(fe)
    # define and compile unsupervised discriminator model
    d_model = Model(in_sample, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    return d_model, c_model

##### plot the Discriminator
# d_model, c_model = define_discriminator()
# plot_model(c_model, to_file='./images/discriminator1_5x5_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(d_model, to_file='./images/discriminator2_5x5_plot.png', show_shapes=True, show_layer_names=True)

# define the standalone generator model
# def define_generator(latent_dim):
#     # sample generator input
#     in_lat = Input(shape=(latent_dim,))
#     # foundation for 5x5 sample
#     n_nodes = 100 * 1
#     gen = Dense(n_nodes)(in_lat)
#     gen = LeakyReLU(alpha=0.2)(gen)
#     gen = Reshape((1, 100))(gen)
#     # upsample to 2x2
#     gen = UpSampling1D(size=25)(gen)
#     gen = Conv1D(100, (3), strides=(5), padding='same')(gen)
#     gen = LeakyReLU(alpha=0.2)(gen)

#     gen = UpSampling1D(size=10)(gen)
#     gen = Conv1D(100, (5), strides=(2), padding='same')(gen)
#     gen = LeakyReLU(alpha=0.2)(gen)
#     # output
#     out_layer = Conv1D(1, (2), activation='tanh', padding='same')(gen)
#     # define model
#     model = Model(in_lat, out_layer)
#     return model

def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 3x4 sample
    n_nodes = 100 * 1
    gen = Dense(n_nodes)(in_lat)
    gen = Dense(50, activation='relu')(gen)
    gen = UpSampling1D(size=2)(gen)
    gen = Dense(100, activation='relu')(gen)
    gen = UpSampling1D(size=2)(gen)
    gen = Dense(25, activation='sigmoid')(gen)

    out_layer = Reshape((25, 1))(gen)
 
    # output
    # define model
    model = Model(in_lat, out_layer)
    return model

##### plot the Generator
# g_model = define_generator(100)
# plot_model(g_model, to_file='./images/generator_010.png', show_shapes=True, show_layer_names=True)