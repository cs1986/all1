from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#get_ipython().run_line_magic('matplotlib', 'inline')

housing = fetch_openml(name="house_prices", as_frame=True)
housing.keys()

data = pd.DataFrame(housing.data, columns=housing.feature_names)
data.head()
data.keys()
data['Price'] = housing.target
data.head()
data.describe()
data.info()

sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.displot(data['Price'], bins=30)
plt.show()

correlation_matrix = data.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

plt.figure(figsize=(20, 5))
features = ['YearBuilt', 'GarageArea']
target = data['Price']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = data[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


x = pd.DataFrame(np.c_[data['YearBuilt'], data['GarageArea']], columns=[ 'YearBuilt', 'GarageArea'])
y = data['Price']


X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=5)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, sep='\n')

model = LinearRegression()
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_pred)))
r2 = r2_score(Y_test, y_pred)
print('Model Preformance', f'RMSE is {rmse}', f'R2 score is {r2}', sep='\n')

sample_data = [[1990, 9000]]
price = model.predict(sample_data)
print(price[0])
