from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataset = pd.read_csv(
    'https://raw.githubusercontent.com/mk-gurucharan/Classification/master/IrisDataset.csv')

X = dataset.iloc[:, :4].values
y = dataset['species'].values
print(dataset.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Accuracy : ", accuracy_score(y_test, y_pred))
print(cm)

df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df)

m = accuracy_score(y_test, y_pred)

print("error rate:-", 1-m)
print('Precision:', precision_score(y_test, y_pred, average='micro'))
print("Recall Score:", recall_score(y_test, y_pred, average='micro'))
