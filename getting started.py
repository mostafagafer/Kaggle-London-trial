import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


train = pd.read_csv('C:/Users/Mostafa_2/Desktop/udemy python/Kaggle/getting started/train.csv')
train.head()
train.shape

train_label = pd.read_csv(
    'C:/Users/Mostafa_2/Desktop/udemy python/Kaggle/getting started/trainLabels.csv')
train_label.head()
train_label.shape

test = pd.read_csv('C:/Users/Mostafa_2/Desktop/udemy python/Kaggle/getting started/test.csv')
test.head()
test.shape


###########################
# logesticRegreesion

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# assign X and y to train and label
X = train
y = train_label


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg.score(x_train, y_train)  # .83
log_reg.score(x_test, y_test)  # .81

log_reg100 = LogisticRegression(C=100)
log_reg100.fit(x_train, y_train)
log_reg100.score(x_train, y_train)  # .83
log_reg100.score(x_test, y_test)  # .81


log_reg001 = LogisticRegression(C=0.00100)
log_reg001.fit(x_train, y_train)
log_reg001.score(x_train, y_train)  # .80
log_reg001.score(x_test, y_test)  # .81


###########################################
from sklearn.neighbors import KNeighborsClassifier


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn.score(x_train, y_train)  # .938
knn.score(x_test, y_test)  # .895

training_accuracy = []
test_accuracy = []
neighbors_setting = range(1, 30)
for n_neighbor in neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors=n_neighbor)
    clf.fit(x_train, y_train)
    training_accuracy.append(clf.score(x_train, y_train))
    test_accuracy.append(clf.score(x_test, y_test))

plt.plot(neighbors_setting, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_setting, test_accuracy, label='Accuracy of the test set')


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
knn.score(x_train, y_train)  # .938
knn.score(x_test, y_test)  # .9075
answer = knn.predict(test)
answer
answer.shape

df = pd.DataFrame(answer, columns=["label"])
df
df.shape
test.head()
test.shape

result = pd.concat([test, df], axis=1, sort=False)
result
result.to_csv('reultLondon.csv')

# testin the accuracy to the 9000 test.csv
X = df
X
X.shape
y = test
y
y.shape

# to make sure form the accuracy

from sklearn import metrics

X = np.array(result.drop(['label'], 1))
y = np.array(df['label'])

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))
