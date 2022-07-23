
# Import packages, jika ada yang ingin disesuaikan silakan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

#Ignore warnings
import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning)
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning)

PATH = '/content/drive/MyDrive/Colab Notebooks/Tugas 8 Wrangling/data_8/'

from google.colab import drive
drive.mount('/content/drive')

# Import data, house price
df = pd.read_csv(PATH+'train.csv')
df.head()

# Cek kolom nama.
df.columns.values

total_null = df.isna().sum().sort_values(ascending = False) 
percentage_null = ((total_null/df.isna().count()*100).sort_values(ascending = False))

missing_data = pd.concat([total_null, percentage_null], axis = 1, keys = ['Total', 'Percentage'])
missing_data.head(20)

df = df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)

# Cek jumlah kolom setelah dibuang
len(df.columns.values)

total_null = df.isna().sum().sort_values(ascending = False) 
total_null.head(15)

missing_columns = ['LotFrontage', 'GarageFinish', 'GarageType', 'GarageCond', 'GarageQual', 'GarageYrBlt', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'MasVnrType', 'MasVnrArea', 'Electrical']
missing_columns

# Untuk mengetahui distribusi data dari data missing_columns, lakukan visualisasi, JANGAN DIUBAH!
from collections import Counter

def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        try:
            df[var_name].hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (8, 24))
        except:
            cnt = Counter(df[var_name])
            cnt.hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (8, 24))
        ax.set_title(var_name, fontsize = 10)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
        ax.set_xlabel('')
    fig.tight_layout(rect = [0, 0.03, 1, 0.95])  # Improves appearance a bit.
    plt.show()

draw_histograms(df, missing_columns, 7, 2)

df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)


df['GarageFinish'].fillna(df['GarageFinish'].mode()[0], inplace=True)
df['GarageType'].fillna(df['GarageType'].mode()[0], inplace=True)
df['GarageCond'].fillna(df['GarageCond'].mode()[0], inplace=True)
df['GarageQual'].fillna(df['GarageQual'].mode()[0], inplace=True)
df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0], inplace=True)
df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)
df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace=True)
df['BsmtCond'].fillna(df['BsmtCond'].mode()[0], inplace=True)
df['BsmtQual'].fillna(df['BsmtQual'].mode()[0], inplace=True)
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)

df.isna().sum().sort_values(ascending = False)

numerical_columns = df._get_numeric_data().columns
numerical_df = df[numerical_columns]

# Set kolom 'Id' sebagai indeks
numerical_df.set_index(['Id'], inplace=True)
numerical_df.head()

categorical_df = df.drop(numerical_columns, axis=1)
categorical_df.head()

# Distribusi sebelum
sns.distplot(numerical_df['1stFlrSF'])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(numerical_df[['1stFlrSF']])
numerical_df['1stFlrSF_scaled_std'] = scaler.transform(numerical_df[['1stFlrSF']]) # Membuat nama kolom baru yang sudah dilakukan Standard Scaling

# Distribusi sesudah
sns.distplot(numerical_df['1stFlrSF_scaled_std'])


# Distribusi sebelum
sns.distplot(numerical_df['LotArea'])

numerical_df['LotArea_log_scale'] = np.log(numerical_df['LotArea'])

# Distribusi sesudah
sns.distplot(numerical_df['LotArea_log_scale']) # Membuat kolom baru yang sudah diubah menjadi skala logaritmik

ExterCond_dict = {'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}
categorical_df.replace({'ExterCond':ExterCond_dict}, inplace=True)

categorical_df.ExterCond.head()

# Cek semua nama kolom di tabel numerical_df

numerical_df.columns.values

# Distplot semua kolom numerik, untuk mengetahui distribusi data

f = pd.melt(numerical_df, value_vars=numerical_df.columns)
g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")


# Menghapus kolom-kolom yang berhubungan dengan 'Basement' 

numerical_df = numerical_df.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','BsmtFullBath','BsmtHalfBath'], axis=1)

# Mengecek persentase nilai nol pada kolom 'TotalBsmtSF'

jumlah_nol_TotalBsmtSF = numerical_df['TotalBsmtSF'][numerical_df['TotalBsmtSF'] == 0].count()
jumlah_baris_TotalBsmtSF = numerical_df['TotalBsmtSF'].count()
persentase_nol_TotalBsmtSF = jumlah_nol_TotalBsmtSF/jumlah_baris_TotalBsmtSF*100
persentase_nol_TotalBsmtSF

scaler.fit(numerical_df[['TotalBsmtSF']])
numerical_df['TotalBsmtSF_scaled_std'] = scaler.transform(numerical_df[['TotalBsmtSF']]) # Membuat nama kolom baru yang sudah dilakukan Standard Scaling

scaler.fit(numerical_df[['2ndFlrSF']])
numerical_df['2ndFlrSF_scaled_std'] = scaler.transform(numerical_df[['2ndFlrSF']]) # Membuat nama kolom baru yang sudah dilakukan Standard Scaling

scaler.fit(numerical_df[['GarageArea']])
numerical_df['GarageArea_scaled_std'] = scaler.transform(numerical_df[['GarageArea']]) # Membuat nama kolom baru yang sudah dilakukan Standard Scaling

scaler.fit(numerical_df[['GrLivArea']])
numerical_df['GrLivArea_scaled_std'] = scaler.transform(numerical_df[['GrLivArea']]) # Membuat nama kolom baru yang sudah dilakukan Standard Scaling

# Mengecek persentase nilai nol pada kolom 'PoolArea'

jumlah_nol_PoolArea = numerical_df['PoolArea'][numerical_df['PoolArea'] == 0].count()
jumlah_baris_PoolArea = numerical_df['PoolArea'].count()
persentase_nol_PoolArea = jumlah_nol_PoolArea/jumlah_baris_PoolArea*100
persentase_nol_PoolArea

# Mengecek persentase nilai nol pada kolom 'MasVnrArea'

jumlah_nol_MasVnrArea = numerical_df['MasVnrArea'][numerical_df['MasVnrArea'] == 0].count()
jumlah_baris_MasVnrArea = numerical_df['MasVnrArea'].count()
persentase_nol_MasVnrArea = jumlah_nol_MasVnrArea/jumlah_baris_MasVnrArea*100
persentase_nol_MasVnrArea

# Mengecek persentase nilai nol pada kolom 'LowQualFinSF'

jumlah_nol_LowQualFinSF = numerical_df['LowQualFinSF'][numerical_df['LowQualFinSF'] == 0].count()
jumlah_baris_LowQualFinSF = numerical_df['LowQualFinSF'].count()
persentase_nol_LowQualFinSF = jumlah_nol_LowQualFinSF/jumlah_baris_LowQualFinSF*100
# Mengecek persentase nilai nol pada kolom 'MasVnrArea'persentase_nol_LowQualFinSF

# Mengecek persentase nilai nol pada kolom 'WoodDeckSF'

jumlah_nol_WoodDeckSF = numerical_df['WoodDeckSF'][numerical_df['WoodDeckSF'] == 0].count()
jumlah_baris_WoodDeckSF = numerical_df['WoodDeckSF'].count()
persentase_nol_WoodDeckSF = jumlah_nol_WoodDeckSF/jumlah_baris_WoodDeckSF*100
persentase_nol_WoodDeckSF

# Menghapus kolom 'PoolArea' dan 'LowQualFinSF'

numerical_df = numerical_df.drop(['PoolArea', 'LowQualFinSF'], axis=1)

"""Proses **Standard Scaling** juga akan saya lakukan terhadap kolom 'MasVnrArea' dan 'WoodDeckSF', serupa dengan kolom-kolom lain yang nilainya adalah luas area"""

scaler.fit(numerical_df[['2ndFlrSF']])
numerical_df['2ndFlrSF_scaled_std'] = scaler.transform(numerical_df[['2ndFlrSF']]) # Membuat nama kolom baru yang sudah dilakukan Standard Scaling

scaler.fit(numerical_df[['GarageArea']])
numerical_df['GarageArea_scaled_std'] = scaler.transform(numerical_df[['GarageArea']]) # Membuat nama kolom baru yang sudah dilakukan Standard Scaling

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

minmax.fit(numerical_df[['MSSubClass']])
numerical_df['MSSubClass_scaled_minmax'] = minmax.transform(numerical_df[['MSSubClass']]) # Membuat nama kolom baru yang sudah dilakukan MinMax Scaling

minmax.fit(numerical_df[['OverallQual']])
numerical_df['OverallQual_scaled_minmax'] = minmax.transform(numerical_df[['OverallQual']]) # Membuat nama kolom baru yang sudah dilakukan MinMax Scaling

minmax.fit(numerical_df[['OverallCond']])
numerical_df['OverallCond_scaled_minmax'] = minmax.transform(numerical_df[['OverallCond']]) # Membuat nama kolom baru yang sudah dilakukan MinMax Scaling

# Pengecekan pada kolom 'MiscVal'

print("nilai unik pada kolom 'MiscVal' adalah:", sorted(numerical_df['MiscVal'].unique()))
print("nilai mean pada kolom 'MiscVal' adalah:", numerical_df['MiscVal'].mean())

# Pengecekan pada kolom 'SalePrice'

print("nilai unik pada kolom 'SalePrice' adalah:", sorted(numerical_df['SalePrice'].unique()))
print("nilai mean pada kolom 'SalePrice' adalah:", numerical_df['SalePrice'].mean())

# Menghapus nilai outlier pada kolom 'MiscVal' dan membuat kolom baru

numerical_df['MiscVal_corrected'] = numerical_df['MiscVal'][numerical_df['MiscVal'] <= 54]


# Menghapus nilai outlier pada kolom 'SalePrice' dan membuat kolom baru

numerical_df['SalePrice_corrected'] = numerical_df['SalePrice'][numerical_df['SalePrice'] <= 400000]

Kemudian plot lagi distplot setelah tabel numerical_df dilakukan proses data cleansing.

# Distplot semua kolom numerik, untuk mengetahui distribusi data setelah dilakukan cleansing

f = pd.melt(numerical_df, value_vars=numerical_df.columns)
g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

# Menghapus kolom 'MiscVal' dan 'MiscVal_corrected'

numerical_df = numerical_df.drop(['MiscVal', 'MiscVal_corrected'], axis=1)

numerical_df.reset_index(inplace=True)


categorical_df

# Cek semua nama kolom di tabel categorical_df

categorical_df.columns.values

# Untuk mengetahui distribusi data dari categorical_df

from collections import Counter

def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        try:
            df[var_name].hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (20, 48))
        except:
            cnt = Counter(df[var_name])
            cnt.hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (20, 48))
        ax.set_title(var_name, fontsize = 10)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
        ax.set_xlabel('')
    fig.tight_layout(rect = [0, 0.03, 1, 0.95])  # Improves appearance a bit.
    plt.show()

draw_histograms(df, categorical_df, 19, 2)

Ordinal_dict = {'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}
categorical_df.replace({'ExterQual':Ordinal_dict, 'ExterCond':Ordinal_dict,'BsmtQual':Ordinal_dict, 'BsmtCond':Ordinal_dict, 
                        'HeatingQC':Ordinal_dict, 'KitchenQual':Ordinal_dict, 'GarageQual':Ordinal_dict, 'GarageCond':Ordinal_dict}, 
                       inplace=True)

# Pengecekan nilai unik pada semua kolom ordinal

print("nilai unik pada kolom 'ExterQual' adalah:", sorted(categorical_df['ExterQual'].unique()))
print("nilai unik pada kolom 'ExterCond' adalah:", sorted(categorical_df['ExterCond'].unique()))
print("nilai unik pada kolom 'BsmtQual' adalah:", sorted(categorical_df['BsmtQual'].unique()))
print("nilai unik pada kolom 'BsmtCond' adalah:", sorted(categorical_df['BsmtCond'].unique()))
print("nilai unik pada kolom 'HeatingQC' adalah:", sorted(categorical_df['HeatingQC'].unique()))
print("nilai unik pada kolom 'KitchenQual' adalah:", sorted(categorical_df['KitchenQual'].unique()))
print("nilai unik pada kolom 'GarageQual' adalah:", sorted(categorical_df['GarageQual'].unique()))
print("nilai unik pada kolom 'GarageCond' adalah:", sorted(categorical_df['GarageCond'].unique()))

df_cleansed = pd.concat([numerical_df, categorical_df], axis=1)
df_cleansed.head()

