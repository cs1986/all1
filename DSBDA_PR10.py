import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

df = pd.read_csv('iris.csv')
print(df)

column = len(list(df))
print(column)

df.info()
np.unique(df['Species'])

get_ipython().run_line_magic('matplotlib', 'inline')


fig, axes = plt.subplots(2, 2, figsize=(16, 8))

axes[0, 0].set_title("Distribution of first column")
axes[0, 0].hist(df['SepalLengthCm'])

axes[0, 1].set_title("Distribution of second column")
axes[0, 1].hist(df['SepalWidthCm'])

axes[1, 0].set_title("Distribution of third column")
axes[1, 0].hist(df['PetalLengthCm'])

axes[1, 1].set_title("Distribution of fourth column")
axes[1, 1].hist(df['PetalWidthCm'])


data_to_plot = [df['SepalLengthCm'], df['SepalWidthCm'],
                df['PetalLengthCm'], df['PetalWidthCm']]
sns.set_style('whitegrid')

fig = plt.figure(1, figsize=(12, 8))

ax = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot)

df.describe()
