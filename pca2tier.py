import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df = df.drop(columns=['id', 'partlybad', 'date'])
dftest = pd.read_csv('test.csv')
ids = dftest['id']
dfx = df.drop(columns='class4')
dfy4 = df['class4']
dfy2 = dfy4.map(lambda x : 0 if x == 'nonevent' else 1)

x_events = dfx[(dfy4 != 'nonevent')]
dfy3 = dfy4[(dfy4 != 'nonevent')]

normalized_events = normalize(x_events)
normalized = normalize(dfx)