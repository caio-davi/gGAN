from numpy import loadtxt
from keras.models import load_model
from numpy import ones
from numpy import sum
from numpy import array
from numpy import reshape
from numpy.random import randn
import pandas as pd
import os
import pre_processing
import sys


path = '/gGAN/src/'

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
            except Exception as e:
                print(e)

def invert_dic(dictionary):
    to_list = []
    for key, item in dictionary.items():
        to_list.append(list({v: k for k, v in item.items()}.keys()))
    return to_list; 

 
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

for i in range(len(predict_c)):
    new_sample = pd.DataFrame(data=generated[i])
    if(predict_d[i]>0.5):
        if(predict_c[i]>0.5):
            sd_samples.append(new_sample)
        else:
            df_samples.append(new_sample)

dic = pre_processing.init(path, 'SVM', '1' , dic=True)

inv_map = invert_dic(dic)

#print('Dict: ', inv_map)
print('sample: ', df_samples[0])

sys.exit()

folders = ['./data/synthetic/labeled/DF/', './data/synthetic/labeled/SD/'] 

create_folders(folders)
clear_folders(folders)

for i in range(len(df_samples)):
    df_samples[i].to_csv(folders[0]+str(i)+'.csv', index=False, header = False)

for i in range(len(sd_samples)):
    sd_samples[i].to_csv(folders[1]+str(i)+'.csv', index=False, header = False)