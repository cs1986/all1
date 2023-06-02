from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = sns.load_dataset("titanic")
print(df)

cols = df.columns
print(cols)
df.info()
df.describe()
df.isnull().sum()


labelencoder = LabelEncoder()# Assigning numerical values and storing in another column
df['Sex'] = labelencoder.fit_transform(df['Sex'])
print(df)

sns.boxplot(x=df["sex"],y=df["age"])
sns.boxplot(x=df["sex"],y=df["age"],hue=df["survived"])