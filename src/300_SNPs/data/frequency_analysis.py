import pandas as pd
import sys

# Importing data and removing unnecessary headers
def import_data():
    # From Dengue Paper
    labeled_data = pd.read_csv('labeled.csv', header=0)
    diag = labeled_data['diagnose']
    labeled_data = labeled_data.drop(['ID'], axis=1)
    # From 1000Genomes
    unlabeled_data = pd.read_csv('unlabeled.csv', sep=';',header=0)
    unlabeled_data = unlabeled_data.drop(['Patient ID', 'Population', 'rs7277299', 'Unnamed: 299'], axis=1)
    labeled_data = labeled_data[unlabeled_data.columns]
    return labeled_data , unlabeled_data

# get all the frequencies of each polimorfism's genotype
# return a list of pandas.series
def get_frequencies(data):
    frequencies = []
    for snp in data.columns:
        snp_values = data[snp].value_counts()
        frequencies.append(snp_values)
    return frequencies

def compare_frequencies(dataset_A, dataset_B):
    # get the frequencies counts
    freq_A = get_frequencies(dataset_A)
    freq_B = get_frequencies(dataset_B)
    for i in range(len(dataset_A.columns)):
        # make sure we are comparing the same SNPs
        if(freq_A[i].name == freq_B[i].name):
            return None


labeled_data , unlabeled_data = import_data()
compare_frequencies(labeled_data, unlabeled_data)
      