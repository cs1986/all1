from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from math import exp
import pandas as pd
import numpy as np

data = pd.read_csv("Social_Network_Ads.csv")
data.head()
data.describe()

plt.scatter(data['Age'], data['Purchased'])
plt.xlabel("Age")
plt.ylabel("Purchased")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    data["Age"], data["Purchased"], test_size=0.2)

model = LogisticRegression()
model.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1).ravel())

y_pred_sk = model.predict(X_test.values.reshape(-1, 1))
plt.clf()
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred_sk, c="red")

plt.xlabel("age")
plt.ylabel("purchased")
plt.show()

print(
    f"Accuracy = {model.score(X_test.values.reshape(-1, 1),y_test.values.reshape(-1, 1))}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sk).ravel()
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)


Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy {:0.2f}%:".format(Accuracy))

Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))


Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))

err = (fp + fn)/(tp + tn + fn + fp)
print("Error rate {:0.2f}".format(err))
