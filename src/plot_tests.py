import numpy as np
import pandas as pd
from matplotlib.pylab import plt
from sys import exit

header =['instance','step','labeled_loss','labeled_acc','unlabeled_loss','unlabeled_acc','app_t2','loss_t2','acc_t2']

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
#    for metric in data.columns:
    for i in range(2, len(data.columns)-1):
        plt.ylim([0,1])
        plt.plot(data[data.columns[i]], label=data.columns[i])
        plt.savefig('./run/'+name+data.columns[i]+'.png')
        plt.clf()
    # plt.legend()
    
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
    metrics = data.columns
    for metric in metrics:
        plt.plot([data[metric]])
        plt.savefig(name+'_'+metric+'.png')
        


path = '../backups/Model_SVM/'
file_name = 'test_\'model_SVM\'_2020-02-03T22:38:00.223495'
full_name = path+file_name
data = pd.read_csv(full_name+'.log')

plot_metrics(file_name, data)
