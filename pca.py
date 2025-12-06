import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df = df.drop(columns=['id', 'partlybad', 'date'])
dfx = df.drop(columns='class4')
dfy4 = df['class4']
dfy2 = dfy4.map(lambda x : 0 if x == 'nonevent' else 1)

x_events = dfx[(dfy4 != 'nonevent')]
dfy3 = dfy4[(dfy4 != 'nonevent')]

normalized_events = normalize(x_events)
normalized = normalize(dfx)
cum_pve = []
lrscore2 = []
treescore2 = []
nbscore2 = []

lrscore3 = []
treescore3 = []
nbscore3 = []

lrscore4 = []
treescore4 = []
nbscore4 = []


for i in range(50):
    pca = PCA(n_components=i+1)
    x = pca.fit_transform(normalized)
    x_event = pca.fit_transform(normalized_events)
    lr = LogisticRegression()
    tree = DecisionTreeClassifier()
    nb = GaussianNB()

    lrscore2.append(np.mean(cross_val_score(lr, x, dfy2, cv=10)))
    treescore2.append(np.mean(cross_val_score(tree, x, dfy2, cv=10)))
    nbscore2.append(np.mean(cross_val_score(nb, x, dfy2, cv=10)))

    lrscore4.append(np.mean(cross_val_score(lr, x, dfy4, cv=10)))
    treescore4.append(np.mean(cross_val_score(tree, x, dfy4, cv=10)))
    nbscore4.append(np.mean(cross_val_score(nb, x, dfy4, cv=10)))

    lrscore3.append(np.mean(cross_val_score(lr, x_event, dfy3, cv=10)))
    treescore3.append(np.mean(cross_val_score(tree, x_event, dfy3, cv=10)))
    nbscore3.append(np.mean(cross_val_score(nb, x_event, dfy3, cv=10)))

    #cum_pve.append(sum(pca.explained_variance_ratio_))

plt.plot(nbscore2, color='red')
plt.plot(treescore2, color='blue')
plt.plot(lrscore2, color='green')
plt.show()

plt.plot(nbscore3, color='red')
plt.plot(treescore3, color='blue')
plt.plot(lrscore3, color='green')
plt.show()

plt.plot(nbscore4, color='red')
plt.plot(treescore4, color='blue')
plt.plot(lrscore4, color='green')
plt.show()

result = []
