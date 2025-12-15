import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')
ids = dftest['id']
dftest = dftest.drop(columns=['id', 'partlybad', 'date'])
df = df.drop(columns=['id', 'partlybad', 'date'])
dfx = df.drop(columns='class4')
dfy4 = df['class4']
dfy2 = dfy4.map(lambda x : 0 if x == 'nonevent' else 1)

pca = PCA(20)
normalized = normalize(dfx)

x = pca.fit_transform(normalized)
lda2 = LinearDiscriminantAnalysis()
lda2.fit(x, dfy2)
lda4 = LinearDiscriminantAnalysis()
lda4.fit(x, dfy4)

test_norm = normalize(dftest)
xtest = pca.fit_transform(test_norm)
probs = lda2.predict_proba(xtest)
predictions4 = lda4.predict(xtest)
resultdf = pd.DataFrame({'id' : ids, 'class4': predictions4, 'p': probs[:,1]})
resultdf.to_csv('pcaLDA.csv', index = False)