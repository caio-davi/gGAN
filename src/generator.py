from numpy import loadtxt
from keras.models import load_model
from numpy import ones
from numpy import sum
from numpy.random import randn
 
# load model
# c_model = load_model('../backups/Model_SVM/partial_188000/c_model.h5')
d_model = load_model('../backups/Model_SVM/partial_188000/d_model.h5', compile=False)
g_model = load_model('../backups/Model_SVM/partial_188000/g_model.h5', compile=False)

def generate_latent_points(latent_dim, n_samples):
    z_input = randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

def generate_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    samples = generator.predict(z_input)
    y = ones((n_samples, 1))
    return samples, y

generated, labels = generate_samples(g_model, 100, 20000)

print(generated)
predict_d = d_model.predict(generated, verbose=0)
print(predict_d)

count = sum(predict_d>0.5)
print(count)