import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class TwoTierClassifier(BaseEstimator):
    def __init__(self, binmodel, threemodel, bc, tc):
        self.binmodel = binmodel
        self.threemodel = threemodel
        self.bc = bc
        self.tc = tc

    def fit(self, x, y):
        x2 = PCA(self.bc).fit_transform(x)
        y2 = y.map(lambda e : 0 if e == 'nonevent' else 1)
        x3 = PCA(self.tc).fit_transform(x[(y != 'nonevent')])
        y3 = y[(y != 'nonevent')]
        self.binmodel.fit(x2, y2)
        self.threemodel.fit(x3, y3)
    
    def predict(self, x):
        x2 = PCA(self.bc).fit_transform(x)
        x3 = PCA(self.tc).fit_transform(x)
        res = self.binmodel.predict(x2)
        res = res.astype(str)
        for i, r in enumerate(res):
            if r == '0':
                res[i] = 'nonevent'
            else:
                res[i] = self.threemodel.predict([x3[i]])[0]
        return res

    def probs(self, x):
        x2 = PCA(self.bc).fit_transform(x)
        return self.binmodel.predict_proba(x2)

df = pd.read_csv('train.csv')
df = df.drop(columns=['id', 'partlybad', 'date'])
dfx = df.drop(columns='class4')
dfx = normalize(dfx)
dfy = df['class4']

dftest = pd.read_csv('test.csv')
ids = dftest['id']
dftest = dftest.drop(columns=['id', 'partlybad', 'date'])
xtest = normalize(dftest)

lr1 = LogisticRegression(penalty=None, max_iter=1000)
lr2 = LogisticRegression(penalty=None, max_iter=1000)
nb = GaussianNB()
rf = RandomForestClassifier(random_state=0)

two_tierLR = TwoTierClassifier(lr1, lr2, 12, 9)
lrnb = TwoTierClassifier(lr1, nb, 12, 16)
lrrf = TwoTierClassifier(lr1, rf, 12, 13)

two_tierLR.fit(dfx, dfy)
lrnb.fit(dfx, dfy)
lrrf.fit(dfx, dfy)

print(
    np.mean(cross_val_score(two_tierLR, dfx, dfy, scoring='accuracy', cv=10)),
    np.mean(cross_val_score(lrnb, dfx, dfy, scoring='accuracy', cv=10)),
    np.mean(cross_val_score(lrrf, dfx, dfy, scoring='accuracy', cv=10))
)

""" predicts = two_tierLR.predict(xtest)
probs = two_tierLR.probs(xtest)
lrresultdf = pd.DataFrame({'id' : ids, 'class4': predicts, 'p': probs[:,1]})
lrresultdf.to_csv('2tierLR.csv', index = False)

predicts = lrnb.predict(xtest)
probs = lrnb.probs(xtest)
lrnbresultdf = pd.DataFrame({'id' : ids, 'class4': predicts, 'p': probs[:,1]})
lrnbresultdf.to_csv('2tierLRNB.csv', index = False)

predicts = lrrf.predict(xtest)
probs = lrrf.probs(xtest)
lrnbresultdf = pd.DataFrame({'id' : ids, 'class4': predicts, 'p': probs[:,1]})
lrnbresultdf.to_csv('2tierLRRF.csv', index = False) """