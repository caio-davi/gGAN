import pandas as pd
from sklearn import preprocessing

# Importing data and removing unnecessary headers
df = pd.read_csv('dataset.csv', sep='	', header=0)
df = df.drop(['#CHROM', 'POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'], axis=1)
df = df.set_index('ID')
df = df.iloc[:,0:2000]
df = df.transpose()

## Transforming into numeric data. Here we could also use oneHotEncode
for feature in df.columns:
    df_coded = df 
    setattr(df_coded, feature, getattr(df,feature).astype("category").cat.codes)

## Normalizing the data
x = df.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

lim_inf = 0 
lim_sup = 0
i = 1

# Saving in 200 files with 10 samples each one
for x in range(10, 2001, 10):
    lim_sup = x
    df1 = df.iloc[lim_inf:lim_sup,]
    df1.to_csv('./real/sample_'+str(i)+'.csv', index=False, header=False)
    lim_inf = lim_sup
    i += 1

