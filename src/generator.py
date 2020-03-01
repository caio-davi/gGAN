from numpy import loadtxt
from keras.models import load_model
from numpy import ones
from numpy import sum
from numpy import array
from numpy import reshape
from numpy.random import randn
import pandas as pd
import os


def create_folders(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
            
def clear_folders(folders):
    for folder in folders:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
 
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
            new_sample = pd.DataFrame(data=generated[i])
            sd_samples.append(new_sample)
        else:
            new_sample = pd.DataFrame(data=generated[i])
            df_samples.append(new_sample)
    i = i +1

folders = ['./data/synthetic/labeled/DF', './data/synthetic/labeled/SD'] 

create_folders(folders)
clear_folders(folders)

for i in range(len(df_samples)):
    df_samples[i].to_csv(folders[0]+str(i)+'.csv', index=False, header = False)

for i in range(len(sd_samples)):
    sd_samples[i].to_csv(folders[1]+str(i)+'.csv', index=False, header = False)