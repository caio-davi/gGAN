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


path = '/gGAN/'

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

def translate_gen(gen, labels):
    result = labels[0]
    for value in labels:
        if value > gen:
            return result
        else:
            result = value
    return 1

def translate_sample(sample, dictionary):
    translated = []
    for i in range(len(sample)):
        val = translate_gen(sample[i], dictionary[i])
        translated.append(val)
    return translated

def translate_all(samples, dictionary):
    new_samples = []
    for sample in samples:
        new_samples.append(translate_sample(sample, dictionary))
    return new_samples

def generate_latent_points(latent_dim, n_samples):
    z_input = randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

def generate_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    samples = generator.predict(z_input)
    return samples

# load model
c_model = load_model(path + 'backups/Model_SVM/c_model.h5')
d_model = load_model(path + 'backups/Model_SVM/d_model.h5', compile=False)
g_model = load_model(path + 'backups/Model_SVM/g_model.h5', compile=False)

generated = generate_samples(g_model, 100, 100000)

predict_d = d_model.predict(generated, verbose=0)
mask = predict_d>0.5
synthetic_samples = generated[reshape(mask,(generated.shape[0]))]
predict_c = c_model.predict(synthetic_samples   , verbose=0)

dic = pre_processing.init(path, 'SVM', '1' , dic=True)
dic = invert_dic(dic)

df_samples = []
sd_samples = []   

for i in range(len(predict_c)):
    new_sample = pd.DataFrame(data=translate_sample(generated[i], dic))
    if(predict_c[i]>0.5):
        sd_samples.append(new_sample)
    else:
        df_samples.append(new_sample)

folders_names = ['data/synthetic/labeled/DF/','data/synthetic/labeled/SD/', 'data/synthetic/unlabeled'] 
folders = map(lambda folder_name: path + folder_name, folders_names)

create_folders(folders)
clear_folders(folders)

df_sammples_size = len(df_samples)
sd_sammples_size = len(sd_samples)

print('DF Size: ', df_sammples_size)
print('SD Size: ', sd_sammples_size)

for i in range(df_sammples_size):
    df_samples[i].to_csv(folders[0]+str(i)+'.csv', index=False, header = False)
    df_samples[i].to_csv(folders[2]+str(i)+'.csv', index=False, header = False)

for i in range(sd_sammples_size):
    sd_samples[i].to_csv(folders[1]+str(i)+'.csv', index=False, header = False)
    sd_samples[i].to_csv(folders[2]+str(df_sammples_size+i)+'.csv', index=False, header = False)
