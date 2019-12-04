import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os, shutil
from sys import exit 

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
def normalize_data(dataFrame):
    for feature in dataFrame.columns:
        setattr(dataFrame, feature, getattr(dataFrame,feature).astype("category").cat.codes)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(dataFrame.values)
    return pd.DataFrame(x_scaled)

# Normilize data and create a dict for mapping both of the datasets
def create_dict(dataframe_1, dataframe_2):
    dictionary = dict()
    for feature in dataframe_1.columns:
        categories_1 = dataframe_1.loc[:,feature].astype("category").cat.categories
        categories_2 = dataframe_2.loc[:,feature].astype("category").cat.categories
        categories = list(set(categories_1) | set(categories_2))
        values = create_values(len(categories))
        dictionary[feature] = dict(zip(categories,values))
    return dictionary

# Create a fixed sparse array between 0 and 1
def create_values(quantity):
    values = []
    count = 0 
    if (quantity==1):
        return [0.11]
    else:
        space = 1 / (quantity - 1)
    for i in range(quantity):
        values.append(count)
        count += space
    return values

# Map categorical dataframe into normalized numeric values following a dictionary
def map_dataframe(dataframe, dictionary):
    print("Mapping DataFrame...")
    df_coded = dataframe
    num_samples = len(dataframe)
    for feature in dataframe.columns:
        for i in range(num_samples):
            df_coded[feature][i] = dictionary[feature][df_coded[feature][i]]
    return df_coded
    
# Dummy way to get the 2 biggest factors of a number. Just work for n>1
def get_factors(number):
    factors = [] 
    for i in range(1, number):
        if number % i == 0:
            factors.append(i)
    if(len(factors)>3):
        x1 = factors[int(len(factors)/2)+1]
    else:
        x1 = factors[-1]
    x2 = number / x1
    return int(x2), x1

# Create a list with all numbers up to the parameter
def list_to(number):
    l = []
    for i in range(1, number+1):
        l.append(i)
    return l

# Create a matrix of one sample of the dataset. 
def create_matrix(sample):
    x1, x2 = get_factors(len(sample))
    lim_inf = 0
    lim_sup = 0
    cols = list_to(x2)
    df = pd.DataFrame(columns=cols)
    for i in range(x2,len(sample)+1,x2):
        lim_sup = i
        s = sample.iloc[lim_inf:lim_sup]
        s.index = cols
        df = df.append(s)
        lim_inf = lim_sup
    return df

# Create a image of a matrix (it must be normalized!) 
def create_image(data):
    fig = plt.figure(figsize = (10,10)) 
    img = fig.add_subplot(111)
    img.imshow(data.values, cmap='viridis')
    plt.savefig('../images/real_sample.png')

# Create folder with our unlabeled data from 1000Genomes
def create_unlabeled_db(unlabeled_data, dic):
#    df = normilize_data(unlabeled_data)
    df = map_dataframe(unlabeled_data, dic)
    print("Generating Sample CSV files...")
    for i in range(0,len(df.index)):
        new = create_matrix(df.loc[i,])
        new.to_csv('./unlabeled/sample_'+str(i)+'.csv', index=False)
    
def create_labeled_db(labeled_data, dic):
#    df = normilize_data(labeled_data)
    df = map_dataframe(labeled_data, dic)
    print("Generating Sample CSV files...")
    for i in range(0,len(df.index)):
        if(diag[i] == 'DF'):
            # create a matrix from df.loc since it gets the ith row
            new = create_matrix(df.loc[i,])
            new.to_csv('./labeled/DF/sample_'+str(i)+'.csv', index=False)
        if(diag[i] == 'SD'):
            new = create_matrix(df.loc[i,])
            new.to_csv('./labeled/SD/sample_'+str(i)+'.csv', index=False)

def create_split_labeled_db(test_size=0.15):
    clear_folders()
    df = normalize_data(labeled_data)
    half_test_size = int((len(df.index)*test_size/2))
    test_count = [half_test_size, half_test_size]
    print("Generating Sample CSV files...")    
    for i in range(0,len(df.index)):
        if(diag[i] == 'DF'):
            new = create_matrix(df.loc[i,])
            if(test_count[0]>0):
                new.to_csv('./labeled/test/DF/sample_'+str(i)+'.csv')
                test_count[0] -= 1
            else:
                new.to_csv('./labeled/training/DF/sample_'+str(i)+'.csv')
        if(diag[i] == 'SD'):
            new = create_matrix(df.loc[i,])
            if(test_count[1]>0):
                new.to_csv('./labeled/test/SD/sample_'+str(i)+'.csv')
                test_count[1] -= 1
            else:
                new.to_csv('./labeled/training/SD/sample_'+str(i)+'.csv')

def create_folders():
    os.makedirs('labeled/training/SD', exist_ok=True)
    os.makedirs('labeled/training/DF', exist_ok=True)
    os.makedirs('labeled/test/SD', exist_ok=True)
    os.makedirs('labeled/test/DF', exist_ok=True)
    os.makedirs('labeled/DF', exist_ok=True)
    os.makedirs('labeled/SD', exist_ok=True)
    os.makedirs('unlabeled', exist_ok=True)
            
def clear_folders():
    folders = ['./labeled/test/DF','./labeled/test/SD', './labeled/training/DF', './labeled/training/SD' ] 
    for folder in folders:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

# From Dengue Paper
labeled_data = pd.read_csv('labeled.csv', header=0)
diag = labeled_data['diagnose']
labeled_data = labeled_data.drop(['ID'], axis=1)

# From 1000Genomes
unlabeled_data = pd.read_csv('unlabeled.csv', sep=';',header=0)
unlabeled_data = unlabeled_data.drop(['Patient ID', 'Population', 'rs7277299', 'Unnamed: 299'], axis=1)

labeled_data = labeled_data[unlabeled_data.columns]

MAX_DIFF = 0.21
# mask = pd.read_csv('./masks/max_diff_'+str(MAX_DIFF)+'.csv', index_col=None, header=None)
mask = pd.read_csv('./masks/dengue_paper.csv', index_col=None, header=None)

labeled_data = labeled_data[mask[1].tolist()]
unlabeled_data = unlabeled_data[mask[1].tolist()]

dic = create_dict(labeled_data, unlabeled_data)

# clear folders and recreate them
clear_folders()
create_folders()

print("Creating Labeled Sample Data...")
create_labeled_db(labeled_data, dic)
print("Creating Unlabeled Sample Data...")
create_unlabeled_db(unlabeled_data, dic)
print("Finished Pre-Processing Data")
