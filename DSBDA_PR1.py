import pandas as pd
from sklearn import preprocessing
df = pd.read_csv('iris.csv')

print(df.head(10))

print(df.isna().sum())

print(df.dtypes)

df['PetalLengthCm'] = df['PetalLengthCm'].astype("int")
print(df.dtypes)

df['PetalLengthCm'] = df['PetalLengthCm'].astype("int")
print(df.dtypes)

x = df[['SepalLengthCm']].values.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_normalized = pd.DataFrame(x_scaled)

print(df_normalized.head(10))
print(df['Species'].unique())

label_encoder = preprocessing.LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

print(df)
print(df['Species'].unique())

features_df = df.drop(columns=['Species'])
enc = preprocessing.OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(df[['Species']]))
df_encode = features_df.join(enc_df)
df_encode.rename(columns={0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-virginica'}, inplace=True)

print(df_encode.tail(40))

one_hot_df = pd.get_dummies(df, prefix="Species",columns=['Species'], drop_first=False)

print(one_hot_df.head(20), one_hot_df.tail(20))
