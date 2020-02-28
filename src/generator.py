from numpy import loadtxt
from keras.models import load_model
from numpy import ones
from numpy import sum
from numpy import array
from numpy import reshape
from numpy.random import randn
import os
 
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
mask = predict_d>0.5
synthetic_samples = generated[reshape(mask,(generated.shape[0]))]
predict_c = c_model.predict(synthetic_samples   , verbose=0)

print("Total Real-like: ",sum(predict_d>0.5))
print("Total Severe: ",sum(predict_c>0.5))

df_samples = []
sd_samples = []
i = 0
while len(sd_samples) < 500:
    if(predict_d[i]>0.5):
        if(predict_c[i]>0.5):
            sd_samples.append(generated[i])
        else:
            df_samples.append(generated[i])
    i = i +1

print(len(sd_samples))
print(len(df_samples))


os.makedirs('./data/synthetic_data/labeled/DF', exist_ok=True)
os.makedirs('./data/synthetic_data/labeled/SD', exist_ok=True)