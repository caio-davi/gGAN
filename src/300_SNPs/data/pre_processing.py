import pandas as pd
import sys
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Importing data and removing unnecessary headers
labeled_data = pd.read_csv('labeled.csv', header=0)
labeled_data = labeled_data.drop(['ID', 'diagnose'], axis=1)

unlabeled_data = pd.read_csv('unlabeled.csv', sep=';',header=0)
unlabeled_data = unlabeled_data.drop(['Patient ID', 'Population', 'rs7277299', 'Unnamed: 299'], axis=1)

labeled_data = labeled_data[unlabeled_data.columns]

# List the most frequent genotype for each SNP and their frequency
def most_frequents(data):
    names = []
    allele_1 = []
    comp = []
    for snp in data.columns:
        snp_values = data[snp].value_counts()
        names.append(snp_values.name)
        allele_1.append(snp_values.index[0])
        comp.append(snp_values[0]/len(data))
    data = {'allele' : allele_1, 'freq':comp}
    return pd.DataFrame(data, index=names, columns=['allele', 'freq'])

# Compare the different encodes by looking the most frequent genotype of each SNP
def compare_encodes(code_A, code_B, max_diff=None):
    snps = []
    diff = []
    a = [] 
    b = []
    c = []
    d = []
    for snp in code_A.index:
        if(code_A.loc[snp,:][0] != code_B.loc[snp,:][0]):
            snps.append(snp)
            if(max_diff and abs(code_A.loc[snp,:][1] - code_B.loc[snp,:][1]) > max_diff):
                diff.append(snp)
                a.append(code_A.loc[snp,:][0])
                b.append(code_A.loc[snp,:][1])
                c.append(code_B.loc[snp,:][0])
                d.append(code_B.loc[snp,:][1])
    data = {'code_a': a, 'freq_a': b , 'code_b' : c, 'freq_b': d}
    diff = pd.DataFrame(data, index=diff, columns=['code_a','freq_a','code_b','freq_b'])
    return diff if (max_diff) else snps

# This will normilize the data, but I'm not sure if I can make the way back
# Since we want to generate new datasets, we will need to do it
# Therefore, we have to rewrite this.
def normilize_data(dataFrame):
    for feature in dataFrame.columns:
        df_coded = dataFrame 
        setattr(df_coded, feature, getattr(dataFrame,feature).astype("category").cat.codes)
    x = dataFrame.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dataFrame = pd.DataFrame(x_scaled)
    return dataFrame

# Create a image of one sample of the dataset. 
def create_image(sample, sample_row):
    lim_inf = 0 
    lim_sup = 0
    data = []
    cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    for i in range(21,294,21):
        lim_sup = i
        df = sample.iloc[sample_row:sample_row+1,lim_inf: lim_sup]
        df.columns = cols
        data.append(df)
        lim_inf = lim_sup
        
    data = pd.concat(data)

    fig = plt.figure(figsize = (10,10)) 
    img = fig.add_subplot(111)
    img.imshow(data.values, cmap='viridis')
    plt.savefig('../images/real_sample_'+str(sample_row)+'.png')


# labeled_code = most_frequents(labeled_data)
# unlabeled_code = most_frequents(unlabeled_data)
# diff = compare_encodes(labeled_code, unlabeled_code, 0.15)
create_image(normilize_data(labeled_data), 5)