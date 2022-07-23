# HousePriceEDA_Bimo

Bismilaahirahmaanirrahiim...

Hello there!

Thank for visiting this page, this is my first data analytics portfolio.
I did a simple python code-writing for exploratory data analysis on the House Price dataset, file train.csv located in "The Datasets" folder in this repository.

This work is inspired by my final assignment in the Data Wrangling course at [Pacmann Academy](https://pacmann.io/), I got a 100/100 mark on this assignment.
I enjoyed my time while I was working on the assignment, even though I also did the additional problem, so I revisited it to improve the problem comprehension, polish the code-writing, and rewrite it in English.

This repository consists of three folders, the original work is labeled as r.-IdfN-wrangling-8, and the latest work is HousePriceEDA_Bimo, so I can document my progress in working on the same dataset with different problem approach.
I practice a lot of the course material in working on this notebook, especially from the Data Wrangling and Data Visualization courses.

The most notable progress in my code is how I rewrite the code more efficiently so it can be written in fewer lines of code.
For example, this is the code I wrote in the original work
  ```
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
  ```
I polish it to this lines of code
  ```
  # Creating the list of numerical values
  num_list = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']

  # Creating the list of categorical values
  cat_list = [x for x in missing_columns if x not in num_list]

  # Impute the null values in categorical data with its mode value
  for cat_name in cat_list:
    df[cat_name].fillna(df[cat_name].mode()[0], inplace=True)
  ```

I am not implemented the data transformation (i.e. scaling or normalization) in my latest work as it was assigned in the original one, because I do not sense any urgency to do the transformation while cleansing the dataset.

I highly appreciate your questions, suggestions, and ideas regarding this work.
Let's discuss it in this link -> [My LinkedIn post](https://www.linkedin.com/posts/anggoro-bimo_github-anggoro-bimohousepriceedabimo-activity-6956544417474256896-uzDG?utm_source=linkedin_share&utm_medium=member_desktop_web)

Thank you very much for your kind attention.
