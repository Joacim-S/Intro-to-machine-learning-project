import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')
ids = dftest['id']
dftest = dftest.drop(columns=['id', 'partlybad', 'date'])
df = df.drop(columns=['id', 'partlybad', 'date'])
dfx = df.drop(columns='class4')
dfy4 = df['class4']

pca = PCA(9)
normalized = normalize(dfx)

x = pca.fit_transform(normalized)
nb2 = GaussianNB()
nb4 = GaussianNB()
nb4.fit(x, dfy4)

test_norm = normalize(dftest)
xtest = pca.fit_transform(test_norm)
probs = nb4.predict_proba(xtest)
predictions4 = nb4.predict(xtest)
resultdf = pd.DataFrame({'id' : ids, 'class4': predictions4, 'p': probs[:,1]})
resultdf.to_csv('pcaGNB.csv', index = False)


pca11 = PCA(12)

x2 = pca.fit_transform(normalized)
lr = LogisticRegression(penalty=None, max_iter=10000)
lr.fit(x2, dfy4)

probs = lr.predict_proba(xtest)
predictions4 = lr.predict(xtest)
resultdf = pd.DataFrame({'id' : ids, 'class4': predictions4, 'p': probs[:,1]})
resultdf.to_csv('pcaLR.csv', index = False)
