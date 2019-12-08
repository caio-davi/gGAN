import numpy as np
import pandas as pd
from matplotlib.pylab import plt

def split_instances(data):
    instances = []
    index = 0
    instances.append([])
    for i in range(len(data)):
        if (data.at[i,'instance'] == index+1):
            instances[index].append(data.at[i,'labeled_loss'])
        else:
            index += 1
            instances.append([])
            instances[index].append(data.at[i,'labeled_loss'])
    return instances

#Just for one instance
def plot_metrics(name, data):
    metrics = ['labeled_loss','labeled_acc','unlabeled_loss','unlabeled_acc']
    for metric in metrics:
        plt.plot(data[metric], label=metric)
    plt.legend()
    plt.savefig(name+'.png')
    
def pretty_plot_metrics(name, data):
    plt.plot(data['labeled_acc'], '-', color='b', label='Labeled Accuracy')
    plt.plot(data['labeled_loss'], ':', color='b',label='Labeled Loss')
    plt.plot(data['unlabeled_acc'], '-', color='r', label='Unsupervised Accuracy')
    plt.plot(data['unlabeled_loss'], ':', color='r', label='Unsupervised Loss')
    plt.legend()
    plt.savefig(name+'.png')
    
def pretty_plot_compare3_acc(name):
    path = './run/'
    file_name = 'model_007'
    data1 = pd.read_csv(path + file_name + '.log')
    file_name = 'model_010'
    data2 = pd.read_csv(path + file_name + '.log')
    file_name = 'model_021'
    data3 = pd.read_csv(path + file_name + '.log')
    metrics = ['labeled_loss','labeled_acc','unlabeled_loss','unlabeled_acc']
    plt.plot(data1['labeled_acc'], ':', color='b', label='Alelic Frequency 0.01 Labeled Accuracy')
    plt.plot(data2['labeled_acc'], ':', color='r', label='Alelic Frequency 0.10 Labeled Accuracy')
    plt.plot(data3['labeled_acc'], ':', color='g', label='Alelic Frequency 0.21 Labeled Accuracy')
    plt.plot(data1['unlabeled_acc'], '-', color='b', label='Alelic Frequency 0.01 Unsupervised Accuracy')
    plt.plot(data2['unlabeled_acc'], '-', color='r', label='Alelic Frequency 0.10 Unsupervised Accuracy')
    plt.plot(data3['unlabeled_acc'], '-', color='g', label='Alelic Frequency 0.21 Unsupervised Accuracy')
    # plt.legend()
    plt.savefig(name+'.png')
    
def pretty_plot_compare3_loss(name):
    path = './run/'
    file_name = 'model_007'
    data1 = pd.read_csv(path + file_name + '.log')
    file_name = 'model_010'
    data2 = pd.read_csv(path + file_name + '.log')
    file_name = 'model_021'
    data3 = pd.read_csv(path + file_name + '.log')
    metrics = ['labeled_loss','labeled_acc','unlabeled_loss','unlabeled_acc']
    plt.plot(data1['labeled_loss'], ':', color='b', label='Alelic Frequency 0.01 Labeled Loss')
    plt.plot(data2['labeled_loss'], ':', color='r', label='Alelic Frequency 0.10 Labeled Loss')
    plt.plot(data3['labeled_loss'], ':', color='g', label='Alelic Frequency 0.21 Labeled Loss')
    plt.plot(data1['unlabeled_loss'], '-', color='b', label='Alelic Frequency 0.01 Unsupervised Loss')
    plt.plot(data2['unlabeled_loss'], '-', color='r', label='Alelic Frequency 0.10 Unsupervised Loss')
    plt.plot(data3['unlabeled_loss'], '-', color='g', label='Alelic Frequency 0.21 Unsupervised Loss')
    # plt.legend()
    plt.savefig(name+'.png')

def save_plot(name, data):
    instances = split_instances(data)
    for i in range(len(instances)):
        plt.plot(instances[i])
    plt.savefig(name+'.png')

path = './run/'
file_name = 'dengue_paper'
full_name = path+file_name
data = pd.read_csv(full_name+'.log')


# save_plot(full_name, data)
# pretty_plot_metrics(full_name, data)
# pretty_plot_compare3_acc('comparing_acc')
pretty_plot_compare3_loss('comparing_loss')