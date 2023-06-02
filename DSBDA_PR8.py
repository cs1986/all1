
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = sns.load_dataset('titanic')
dataset.head()

sns.displot(dataset['fare'])
sns.displot(dataset['fare'], kde=False, bins=10)
sns.jointplot(x='age', y='fare', data=dataset)
sns.jointplot(x='age', y='fare', data=dataset, kind='hex')
sns.pairplot(dataset)

dataset = dataset.dropna()
sns.pairplot(dataset, hue='sex')
sns.rugplot(dataset['fare'])