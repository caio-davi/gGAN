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
        snp_values = snp_values / snp_values.sum()
        frequencies.append(snp_values)
    return frequencies

def get_allele_frequency(allele, snp_frequencies):
    for i in range(len(snp_frequencies)):
        if snp_frequencies.index[i] == allele:
            return snp_frequencies[i]

def compare_frequencies(freq_A, freq_B):
    maxDiff = 0
    shared = list(set(freq_A.index) & set(freq_B.index))
    if(len(shared) == 0):
        return 1
    for allele in shared:
        diff = abs(get_allele_frequency(allele, freq_A) - get_allele_frequency(allele, freq_B))
        if (diff>maxDiff):
            maxDiff = diff
    return maxDiff
    
def compare_all_frequencies(dataset_A, dataset_B):
    # get the frequencies counts
    freq_A = get_frequencies(dataset_A)
    freq_B = get_frequencies(dataset_B)
    diffs = []
    for i in range(len(dataset_A.columns)):
        # make sure we are comparing the same SNPs
        if(freq_A[i].name == freq_B[i].name):
            diffs.append(compare_frequencies(freq_A[i], freq_B[i]))
        else:
            raise IndexError('Index error.')
    return diffs 

def create_mask(diff_freq, threshold):
    mask = []
    for i in range(len(diff_freq)):
        if (diff_freq[i] < threshold):
            mask.append(diff_freq.index[i])
    return pd.Series(mask)


MAX_DIFF = 0.07
labeled_data , unlabeled_data = import_data()
diff_freq = compare_all_frequencies(labeled_data, unlabeled_data)
diff_freq = pd.Series(diff_freq, index=labeled_data.columns)
mask = create_mask(diff_freq, MAX_DIFF)
mask.to_csv('./masks/max_diff_'+str(MAX_DIFF)+'.csv')
print(mask)
print(len(mask))
