from numpy import loadtxt
from keras.models import load_model
from numpy import ones
from numpy import sum
from numpy import array
from numpy import reshape
from numpy.random import randn
 
# load model
c_model = load_model('../backups/Model_SVM/c_model.h5')
d_model = load_model('../backups/Model_SVM/d_model.h5', compile=False)
g_model = load_model('../backups/Model_SVM/g_model.h5', compile=False)

def generate_latent_points(latent_dim, n_samples):
    z_input = randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

def generate_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    samples = generator.predict(z_input)
    return samples

generated = generate_samples(g_model, 100, 20000)

predict_d = d_model.predict(generated, verbose=0)
print(type(generated))
print(generated.shape)
mask = predict_d>0.5
print(mask)
synthetic_samples = generated[reshape(mask,(generated.shape[0]))]
predict_c = c_model.predict(synthetic_samples   , verbose=0)
print(len(predict_c))
print(sum(predict_c>0.5))
