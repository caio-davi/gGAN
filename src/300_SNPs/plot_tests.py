import numpy as np
import pandas as pd
from matplotlib.pylab import plt

path = './run/'
file_name = 'test_2019-11-28T18:08:46.501137.log'
full_name = path+file_name
data = pd.read_csv(full_name)

def split_instances(data):
    instances = []
    index = 0
    instances.append([])
    for i in range(len(data)):
        if (data.at[i,'instance'] == index+1):
            instances[index].append(data.at[i,'unlabeled_acc'])
        else:
            index += 1
            instances.append([])
            instances[index].append(data.at[i,'unlabeled_acc'])
    return instances

def save_plot(name, data):
    instances = split_instances(data)
    for i in range(len(instances)):
        plt.plot(instances[i])
    plt.savefig(name+'.png')



save_plot(full_name, data)