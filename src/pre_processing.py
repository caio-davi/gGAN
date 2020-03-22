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
    print("[INFO] Mapping DataFrame...")
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
    plt.savefig(path + 'images/real_sample.png')

# Create unlabeled data 
def create_unlabeled_db(unlabeled_data, dic, dim):
    df = map_dataframe(unlabeled_data, dic)
    print("[INFO] Generating Sample CSV files...")
    for i in range(0,len(df.index)):
        new = df.loc[i,] if dim == 1 else create_matrix(df.loc[i,])
        new.to_csv(path + 'data/unlabeled/sample_'+str(i)+'.csv', index=False, header = False)

# Create labeled data  
def create_labeled_db(labeled_data, diag, dic, dim):
    df = map_dataframe(labeled_data, dic)
    print("[INFO] Generating Sample CSV files...")
    for i in range(0,len(df.index)):
        if(diag[i] == 'DF'):
            new = df.loc[i,] if dim == 1 else create_matrix(df.loc[i,])
            new.to_csv(path + 'data/labeled/DF/sample_'+str(i)+'.csv', index=False, header = False)
        if(diag[i] == 'SD'):
            new = df.loc[i,] if dim == 1 else create_matrix(df.loc[i,])
            new.to_csv(path + 'data/labeled/SD/sample_'+str(i)+'.csv', index=False, header = False)

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

def check_current_sampling(afd):
    f_name = '/gGAN/src/data/current'
    f = open(f_name, "r")
    if afd == f.readline():
        return True
    else:
        return False

def current_sampling(afd):
    current = check_current_sampling(afd)
    if current:
        return True
    else:
        f_name = 'data/current'
        f = open(f_name, "w")
        f.write(afd)
        return False

def init(path, afd, dim, dic=False):

    if (current_sampling(afd) and not dic):
        return

    dim = float(dim)

    # From Dengue Paper
    labeled_data = pd.read_csv(path + 'data/labeled.csv', header=0)
    diag = labeled_data['diagnose']
    labeled_data = labeled_data.drop(['ID'], axis=1)

    # From 1000Genomes
    unlabeled_data = pd.read_csv(path + 'data/unlabeled.csv', sep=';',header=0)
    unlabeled_data = unlabeled_data.drop(['Patient ID', 'Population', 'rs7277299', 'Unnamed: 299'], axis=1)

    labeled_data = labeled_data[unlabeled_data.columns]

    if(str(afd) == 'SVM'):
        mask = pd.read_csv(path + 'data/masks/dengue_paper.csv', index_col=None, header=None)
    elif(str(afd) == 'hybrid'):
        mask = pd.read_csv(path + 'data/masks/hybrid.csv', index_col=None, header=None)
    else:
        mask = pd.read_csv(path + 'data/masks/max_diff_'+str(afd)+'.csv', index_col=None, header=None)

    labeled_data = labeled_data[mask[1].tolist()]
    unlabeled_data = unlabeled_data[mask[1].tolist()]

    dictionary = create_dict(labeled_data, unlabeled_data)
    
    if(dic):
        return dictionary

    # clear folders and recreate them
    # clear_folders() # issue with clearing folders since they may not exist on the first run
    folders = [path+'data/labeled/test/DF',path+'data/labeled/test/SD', path+'data/labeled/training/DF', path+'data/labeled/training/SD' ] 
    create_folders(folders)
    clear_folders(folders)

    print("[INFO] Creating Labeled Sample Data...")
    create_labeled_db(labeled_data, diag, dictionary, dim)
    print("[INFO] Creating Unlabeled Sample Data...")
    create_unlabeled_db(unlabeled_data, dictionary, dim)
    print("[DONE] Finished Pre-Processing Data")