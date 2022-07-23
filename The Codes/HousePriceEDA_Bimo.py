# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Ignore warnings
import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning)
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning)

# Load dataset

from google.colab import drive
drive.mount('/content/drive')

PATH = '/content/drive/MyDrive/Colab Notebooks/Data Wrangling/Week 8/'

df = pd.read_csv(PATH+'train.csv')
df.head()

# Check the features count in the DataFrame
print("Total features in the DataFrame df are:",len(df.columns.values))

# Check the DataFrame information
df.info(verbose = False)

# Create the count of null values in each feature, resulting in series 
total_null = df.isna().sum().sort_values(ascending = False) 

# Create the percentage of null values compared to total rows in each feature, resulting in series
percentage_null = ((total_null/df.isna().count()*100).sort_values(ascending = False))

# Integrate the null values information, resulting in DataFrame
missing_data = pd.concat([total_null, percentage_null], axis = 1, keys = ['Total', 'Percentage'])
missing_data.head(20)

columns_to_be_deleted = missing_data.loc[(missing_data.Percentage > 40)] # Select the target features to be deleted
columns_to_be_deleted = columns_to_be_deleted.index.values.tolist() # Convert the features into the list
columns_to_be_deleted

# Drop the features with >40% null values
df = df.drop(columns_to_be_deleted, axis=1)
missing_data = missing_data.drop(columns_to_be_deleted)

# Recheck the current features count in the DataFrame
print("Total features in the DataFrame df currently are:",len(df.columns.values))

# Recheck the current missing_data DataFrame
missing_data.head(20)

missing_columns = missing_data.loc[(missing_data.Percentage > 0)] # Select the features containing null values
missing_columns = missing_columns.index.values.tolist() # Convert the features into the list
missing_columns


def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        try:
            df[var_name].hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (15, 30))
        except:
            cnt = Counter(df[var_name])
            cnt.hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (8, 24))
        ax.set_title(var_name, fontsize = 10)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
        ax.set_xlabel('')
    fig.tight_layout(rect = [0, 0.03, 1, 0.95])  # Improves appearance a bit.
    plt.show()

draw_histograms(df, missing_columns, 5, 3)


# Creating the list of numerical values
num_list = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
num_list

# Creating the list of categorical values
cat_list = [x for x in missing_columns if x not in num_list]
cat_list

# Impute the null values in numerical data with its mean value
for num_name in num_list:
  df[num_name].fillna(round(df[num_name].mean()), inplace=True)
or the categorical data, I use the **mode value**."""

# Impute the null values in categorical data with its mode value
for cat_name in cat_list:
  df[cat_name].fillna(df[cat_name].mode()[0], inplace=True)

# Recheck the current null values ocurrence
print("Total of null values that occurred in the DataFrame df are:", df.isna().sum().sum())

# Create the numerical data DataFrame
numerical_columns = df._get_numeric_data().columns
numerical_df = df[numerical_columns]
numerical_df = numerical_df.reindex(sorted(numerical_df.columns), axis=1) # Sort the features by name order alphabetically

# Set the 'Id' feature as an index
numerical_df.set_index(['Id'], inplace=True)
numerical_df.head()


# Create the categorical data DataFrame
categorical_df = df.drop(numerical_columns, axis=1)
categorical_df.head()

# Check the DataFrame information
numerical_df.info(verbose = False)

# Recheck the null values ocurrence
print("Total of null values that occurred in the DataFrame numerical_df are:", numerical_df.isna().sum().sum())

# Select the group of secondary basement features
basements = numerical_df.iloc[:,4:9]

# Drop the features
numerical_df.drop(basements, inplace = True, axis = 1)
numerical_df.head()

# Rename the porches type
numerical_df.rename(columns={"3SsnPorch": "Porch3SeasonSF", "EnclosedPorch": "PorchEnclosedSF", "OpenPorchSF": "PorchOpenSF", "ScreenPorch": "PorchScreenSF"}, inplace = True)

# Sort the features by name order alphabetically
numerical_df = numerical_df.reindex(sorted(numerical_df.columns), axis=1)

# Create an assist DataFrame for the porches type features
porches = numerical_df[["Porch3SeasonSF", "PorchEnclosedSF", "PorchOpenSF", "PorchScreenSF"]]

# Create an example DataFrame for porches type comparison
porch_compare = pd.concat([numerical_df[["LotArea"]], porches, numerical_df[["SalePrice"]]], axis = 1)
porch_compare.head(13)

# Rename the area closure features
numerical_df.rename(columns={"GarageArea": "GarageAreaSF", "GrLivArea": "GrLivAreaSF", "LotArea": "LotAreaSF", "LotFrontage": "LotFrontageFeet"
                             , "MasVnrArea": "MasVnrAreaSF", "PoolArea": "PoolAreaSF"}, inplace = True)


# Select the features for bathroom in the basement
bathroom_basement = basements[['BsmtFullBath', 'BsmtHalfBath']]

# Recover the bath_room_basement to numerical_df
numerical_df = numerical_df.join(bathroom_basement)
numerical_df.head()

# Check the bathroom features
numerical_df[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']].head()

# Features summation process

numerical_df['FullBath'] = numerical_df[['FullBath', 'BsmtFullBath']].sum(axis = 1)
numerical_df['HalfBath'] = numerical_df[['HalfBath', 'BsmtHalfBath']].sum(axis = 1)

numerical_df.drop(bathroom_basement, inplace = True, axis = 1) # Drop the bathroom_basement features
numerical_df.rename(columns={"FullBath": "BathroomFullTotal", "HalfBath": "BathroomHalfTotal"}, inplace = True) # Rename the features with the addition of the description "total"
numerical_df = numerical_df.reindex(sorted(numerical_df.columns), axis=1) # Sort the features by name order alphabetically

numerical_df[['BathroomFullTotal', 'BathroomHalfTotal']].head() # Check the bathroom features

# Check the living area features
numerical_df[['GrLivAreaSF', '1stFlrSF', '2ndFlrSF']].head()

# Rename the living area features
numerical_df.rename(columns={"GrLivAreaSF": "LivingAreaTotalSF", "1stFlrSF": "LivingArea1stFlrSF", "2ndFlrSF": "LivingArea2ndFlrSF", "TotalBsmtSF": "BsmtTotalSF"}, inplace = True)

# Create a new feature by summing the existing features
numerical_df['BuildingAreaTotalSF'] = numerical_df[['LivingAreaTotalSF', 'BsmtTotalSF']].sum(axis = 1)

# Check the number of rooms feature
numerical_df[['TotRmsAbvGrd', 'BedroomAbvGr']].head()

# Create a new feature, OtherRooms
numerical_df['OtherRooms'] = numerical_df['TotRmsAbvGrd'] - numerical_df['BedroomAbvGr']

# Rename the bedroom feature
numerical_df.rename(columns={"BedroomAbvGr": "Bedrooms"}, inplace = True)

# Drop the TotRmsAbvGrd feature
numerical_df.drop(['TotRmsAbvGrd'], inplace = True, axis = 1)

# Create assist DataFrames
yearbuilt = numerical_df[["YearBuilt", "YearRemodAdd"]]
area = numerical_df[["LotAreaSF", "BuildingAreaTotalSF", "LivingAreaTotalSF", "LivingArea1stFlrSF", "LivingArea2ndFlrSF", "BsmtTotalSF"]]
rooms = numerical_df[["Bedrooms", "BathroomFullTotal", "BathroomHalfTotal", "OtherRooms", "GarageCars"]]

# Create an example DataFrame for basic features information
basic_info = pd.concat([yearbuilt, area, rooms, numerical_df[["SalePrice"]]], axis = 1)
basic_info.head()

# Distplot the entire features in the numerical_df to inspect the data distribution

f = pd.melt(numerical_df, value_vars=numerical_df.columns)
g = sns.FacetGrid(f, col="variable",  col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

# Calculate the zero values percentage

columns = ['LowQualFinSF', 'MasVnrAreaSF', 'WoodDeckSF', 'PoolAreaSF']
zeros = [len(numerical_df[numerical_df[x] == 0]) for x in  columns] # Count the zero values in certain columns
rows = len(numerical_df.index) # Row count

def percentage(a):
   return a / rows * 100
pct = list(map(percentage, zeros)) # Count the zero values percentage in certain columns

dict = {'feature': columns, 'percentage of zeros': pct} # Integrate the percentage of zero values, resulting in DataFrame
zeros_df = pd.DataFrame(dict)
zeros_df

# Drop the LowQualFinSF feature
numerical_df.drop(['LowQualFinSF'], inplace = True, axis = 1)

# Create a list for selected columns
selected_cols = ['MasVnrAreaSF', 'WoodDeckSF', 'PoolAreaSF']

# Check the features value before conversion
numerical_df[selected_cols].head()

# Check the data type before conversion
numerical_df[selected_cols].dtypes

# Convert the integers into boolean 
numerical_df[selected_cols] = numerical_df[selected_cols].astype(bool)
numerical_df[selected_cols].head()

# Check the data type after the first conversion 
numerical_df[selected_cols].dtypes

# Convert the booleans into "Yes/No" string

binary_dict = {True:'Y', False:'N'} # Create the assist dictionary for the values replacement
numerical_df.replace({'MasVnrAreaSF':binary_dict, 'WoodDeckSF':binary_dict, 'PoolAreaSF':binary_dict}, 
                       inplace=True)
numerical_df[selected_cols].head()

# Check the data type after the second  conversion
numerical_df[selected_cols].dtypes

# Rename the features
numerical_df.rename(columns={"MasVnrAreaSF": "MasonryVnr", "WoodDeckSF": "WoodDeck", "PoolAreaSF": "Pool"}, inplace = True)

# Compare two features by overlaying their plot

plt.figure(figsize=(7, 5))

sns.distplot(numerical_df['OverallCond'])
sns.distplot(numerical_df['OverallQual'])

plt.legend(['OverallCond', 'OverallQual'])
plt.xlabel('Ditribution Comparison')

# Drop the OverallQual feature
numerical_df.drop(['OverallQual'], inplace = True, axis = 1)

# Create a list for selected columns
selected_cols = ['MSSubClass', 'OverallCond']

# Convert the data type
numerical_df[selected_cols] = numerical_df[selected_cols].astype(object)

# Check the data type after the conversion
numerical_df[selected_cols].dtypes

# Move the features to categorical_df

selected_cols = numerical_df[["MasonryVnr", "WoodDeck", "Pool", "MSSubClass", "OverallCond"]].reset_index()
selected_cols.drop(["Id"], inplace = True, axis = 1)
selected_cols # Create an assist DataFrame

numerical_df.drop(["MasonryVnr", "WoodDeck", "Pool", "MSSubClass", "OverallCond"], inplace = True, axis = 1) # Drop the features from numerical_df

categorical_df = categorical_df.join(selected_cols) # Join the selected features to categorical_df


# Distplot the entire features in the numerical_df to inspect the data distribution

f = pd.melt(numerical_df, value_vars=numerical_df.columns)
g = sns.FacetGrid(f, col="variable",  col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")


categorical_df.head()

categorical_df.info()

# Distplot the entire features in the categorical_df to inspect the data distribution

from collections import Counter

def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        try:
            categorical_df[var_name].hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (20, 100))
        except:
            cnt = Counter(df[var_name])
            cnt.hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (20, 100))
        ax.set_title(var_name, fontsize = 10)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
        ax.set_xlabel('')
    fig.tight_layout(rect = [0, 0.03, 1, 0.95])  # Improves appearance a bit.
    plt.show()

draw_histograms(df, categorical_df, 22, 2)

# Rename the feature
categorical_df.rename(columns={"MasVnrType": "MasonryVnrType"}, inplace = True)

# Calculate the mode values percentage

from statistics import mode # Import library

selected_cols = ['Street', 'LandContour', 'Utilities', 'LandSlope', 'Condition2', 'RoofMatl', 'Heating', 'BsmtFinType2', 'Functional', 'GarageQual', 'GarageCond'] 
mode_names = [mode(categorical_df[x]) for x in  selected_cols] # Mode values in every feature in selected_cols
mode_count = [len(categorical_df[x][categorical_df[x] == mode(categorical_df[x])]) for x in  selected_cols] # Count of mode values in every feature in selected_cols
rows = len(categorical_df.index) # Row count

def percentage(a):
   return a / rows * 100
pct = list(map(percentage, mode_count)) # Count the percentage of mode values in certain columns

dict = {'Feature': selected_cols, 'Mode value': mode_names, 'Percentage of mode': pct} # Integrate the percentage of mode values, resulting in DataFrame
mode_df = pd.DataFrame(dict)
mode_df


# Drop the features containing extreme values
categorical_df.drop(categorical_df[selected_cols], inplace = True, axis = 1)

# Reset the index of numerical_df
numerical_df.reset_index(inplace = True)
numerical_df

# Concatenate the DataFrames
df_cleansed = pd.concat([numerical_df, categorical_df], axis=1)
df_cleansed.head()

# Create an assist series
price = df_cleansed['SalePrice']

# Print the descriptive statistics information of `SalePrice` feature
price.describe()

# Print the bigger size of `SalePrice` distplot
plt.figure(figsize=(12,8))
sns.distplot(price)

# Create assist DataFrames
yearbuilt = df_cleansed[["YearBuilt", "YearRemodAdd"]]
area = df_cleansed[["LotAreaSF", "BuildingAreaTotalSF", "LivingAreaTotalSF", "LivingArea1stFlrSF", "LivingArea2ndFlrSF", "BsmtTotalSF"]]
rooms = df_cleansed[["Bedrooms", "BathroomFullTotal", "BathroomHalfTotal", "OtherRooms"]]


# Create an example DataFrame for basic features information
basic_info = pd.concat([yearbuilt, area, rooms, price], axis = 1)
basic_info.head()

# Create the assist DataFrame for the houses >$40,000
outlier_inspection_by_price = basic_info.loc[price > 400000].sort_values('SalePrice')
outlier_inspection_by_price

# Check the comparison of relation between House Price to the Total Building Area and the Lot Area

fig, (ax1, ax2) = plt.subplots(figsize = (20, 8), ncols = 2,sharey = False) 

sns.regplot(x = price, y = df_cleansed['BuildingAreaTotalSF'], ax = ax1, line_kws = {'color': 'purple'}) # Plot for price vs building area
sns.regplot(x = price, y = df_cleansed['LotAreaSF'], ax = ax2, line_kws = {'color': 'purple'}) # Plot for price vs lot area

plt.xlabel('SalePrice')
ax1.set_ylabel('BuildingAreaTotalSF')
ax2.set_ylabel('LotAreaSF')

# Create assist DataFrame for the outliers
outliers = df_cleansed.loc[(df_cleansed.BuildingAreaTotalSF > 6000) & (df_cleansed.SalePrice < 500000) | 
                           (df_cleansed.BuildingAreaTotalSF > 4300) & (df_cleansed.SalePrice < 330000) |
                           (df_cleansed.BuildingAreaTotalSF > 4000) & (df_cleansed.SalePrice < 200000) |
                           (df_cleansed.BuildingAreaTotalSF < 3000) & (df_cleansed.SalePrice > 380000) |
                           (df_cleansed.BuildingAreaTotalSF < 5000) & (df_cleansed.SalePrice > 570000) |
                           (df_cleansed.BuildingAreaTotalSF < 1000) & (df_cleansed.SalePrice < 50000) |
                           (df_cleansed.BuildingAreaTotalSF < 510)]

# Draw the regplot of Total Sale Price vs Building Area with the outliers

import matplotlib.patches as mpatches
plt.figure(figsize = (9, 8))

sns.regplot(x = price, y = df_cleansed['BuildingAreaTotalSF'], line_kws = {'color': 'green'}) # Plot for price vs building area
sns.scatterplot(x = price, y = outliers['BuildingAreaTotalSF'], color = 'red', marker = 's', s = 75) # Plot for outliers

red_dot = mpatches.Patch(color='red', label='Outliers')
plt.legend(handles=[red_dot])

plt.xlabel('SalePrice')
plt.ylabel('BuildingAreaTotalSF')

# Drop the outlier rows
df_cleansed.drop(outliers.index, inplace = True)
df_cleansed.info(verbose = False)

# Check the comparison of relation between House Price to the Total Building Area and the Lot Area

plt.figure(figsize = (9, 8))

sns.regplot(x = df_cleansed['SalePrice'], y = df_cleansed['BuildingAreaTotalSF'], line_kws = {'color': 'green'}) # Plot for price vs building area

plt.xlabel('SalePrice')
plt.ylabel('BuildingAreaTotalSF')

# Save the work and download it to local computer
df_cleansed.to_csv('HousePriceEDA_Bimo.csv', index=False)
from google.colab import files
files.download("HousePriceEDA_Bimo.csv")