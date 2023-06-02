import pandas as pd
import numpy as np
from sklearn import datasets

url = "income.csv"
df = pd.read_csv(url)
df.head()


print(df.age_group.unique())
print(df.income.unique())

df.groupby(df.age_group).count()
df.groupby(df.age_group).min()
df.groupby(df.age_group).max()
df.groupby(df.age_group).mean()
df.groupby(df.age_group).describe()

data = datasets.load_iris()

df2 = pd.DataFrame(data.data, columns=data.feature_names)
df2['species'] = pd.Series(data.target)

df2.head()
df2.species.unique()
df2.groupby(df2.species)["sepal length (cm)"].describe()
df2.groupby(df2.species)["sepal width (cm)"].describe()
df2.groupby(df2.species)["sepal width (cm)"].describe()
df2.groupby(df2.species)["petal length (cm)"].describe()
df2.groupby(df2.species)["petal width (cm)"].describe()
