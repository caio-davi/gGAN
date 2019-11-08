import pandas as pd
import sys
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Importing data and removing unnecessary headers

# From Dengue Paper
labeled_data = pd.read_csv('labeled.csv', header=0)
diag = labeled_data['diagnose']
labeled_data = labeled_data.drop(['ID'], axis=1)

# From 1000Genomes
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
    return pd.DataFrame(x_scaled)

# Create a matrix of one sample of the dataset. 
def create_matrix(sample, sample_row):
    lim_inf = 0 
    lim_sup = 0
    data = []
    cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for i in range(20,241,20):
        lim_sup = i
        df = sample.iloc[sample_row:sample_row+1,lim_inf:lim_sup]
        df.columns = cols
        data.append(df)
        lim_inf = lim_sup
    return pd.concat(data, ignore_index=True)

# Create a image of a matrix (it must be normalized!) 
def create_image(data):
    fig = plt.figure(figsize = (10,10)) 
    img = fig.add_subplot(111)
    img.imshow(data.values, cmap='viridis')
    plt.savefig('../images/real_sample.png')

# Create folder with our unlabeled data from 1000Genomes
def create_unlabeled_db():
    df = normilize_data(unlabeled_data)
    for i in range(0,len(df.index)):
        new = create_matrix(df, i)
        new.to_csv('./unlabeled/sample_'+str(i)+'.csv')
    print('End')
    
def create_labeled_db():
    df = normilize_data(labeled_data)
    df = labeled_data
    test_size = 10
    test_count = [test_size/2, test_size/2]
    for i in range(0,len(df.index)):
        if(diag[i] == 'DF'):
            new = create_matrix(df, i)
            if(test_count[0]>0):
                new.to_csv('./labeled/test/DF/sample_'+str(i)+'.csv')
                test_count[0] -= 1
            new.to_csv('./labeled/training/DF/sample_'+str(i)+'.csv')
        if(diag[i] == 'SD'):
            new = create_matrix(df, i)
            if(test_count[1]>0):
                new.to_csv('./labeled/test/SD/sample_'+str(i)+'.csv')
                test_count[1] -= 1
            new.to_csv('./labeled/training/SD/sample_'+str(i)+'.csv')
            
create_labeled_db()
# sys.exit()