import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_df = test_df.drop(columns=['id', 'date', 'partlybad'])
test_df = (test_df-test_df.mean())/test_df.std()

df = df.drop(columns=['id', 'date', 'partlybad'])

train_x = df.drop(columns=['class4'])
train_x = (train_x-train_x.mean())/train_x.std()
train_y = df['class4']
train_y2 = train_y.copy()
train_x2 = train_x.copy()

#2-tier
train_x = train_x[train_y != 'nonevent']
train_y = train_y[train_y != 'nonevent']

columns = train_x.columns

train_y2 = train_y2.replace('Ia', 'event')
train_y2 = train_y2.replace('Ib', 'event')
train_y2 = train_y2.replace('II', 'event')
    

clf_lasso = LogisticRegression(random_state=0, solver='saga', penalty='l1', C=1, tol=0.001, max_iter=1000)
clf_ridge = LogisticRegression(random_state=0, solver='saga', penalty='l2', C=1, tol=0.001, max_iter=1000)
clf_DT = DecisionTreeClassifier(random_state=0)
clf_RF = RandomForestClassifier(random_state=0)
clf_SVC = SVC(kernel='linear',random_state=0)


t1 = RFECV(clf_lasso, min_features_to_select=17, step=1)
t1.fit(train_x2, train_y2)
test_x2 = t1.transform(test_df)
train_x2 = t1.transform(train_x2)


t2 = RFECV(clf_lasso, min_features_to_select=3, step=1)
t2.fit(train_x, train_y)
test_x = t2.transform(test_df)
train_x = t2.transform(train_x)

clf_lasso = LogisticRegression(random_state=0, solver='saga', penalty='l1', C=1, tol=0.001, max_iter=1000)
clf_lasso2 = LogisticRegression(random_state=0, solver='saga', penalty='l1', C=1, tol=0.001, max_iter=1000)

""" print('train_x')
print(train_x)
print('train_x2')
print(train_x2)
print('train_y')
print(train_y)
print('train_y2')
print(train_y2) """

clf_lasso2.fit(train_x2, train_y2)
clf_lasso.fit(train_x, train_y)

p_bin = clf_lasso2.predict_proba(test_x2)
p_3 = clf_lasso.predict_proba(test_x)

""" print(p_bin)
print(p_3) """
print(clf_lasso2.classes_)
print(clf_lasso.classes_)

labels = []
probs = []

id = range(450, 1415)
for i in range(450, 1415):
  j = 450
  if p_bin[i-j][1] > 0.5:
    labels.append('nonevent')
    probs.append(p_bin[i-j][1])
  elif p_3[i-j][0] > p_3[i-j][1]+p_3[i-j][2]:
    labels.append(clf_lasso.classes_[0])
    probs.append(p_3[i-j][0])
  
  elif p_3[i-j][1] > p_3[i-j][0]+p_3[i-j][2]:
    labels.append(clf_lasso.classes_[1])
    probs.append(p_3[i-j][1])

  else:
    labels.append(clf_lasso.classes_[2])
    probs.append(p_3[i-j][2])

result = pd.DataFrame({'id' : id, 'class4': labels, 'p': probs})
result.to_csv('RFE2tier.csv', index = False)


