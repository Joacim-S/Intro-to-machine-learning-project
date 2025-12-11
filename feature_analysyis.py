import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('train.csv')

df = df.drop(columns=['id', 'date', 'partlybad'])

train_x = df.drop(columns=['class4'])
train_x = (train_x-train_x.mean())/train_x.std()

train_y = df['class4']

df = train_x.copy()
df['class4'] = train_y
train_y2 = train_y.copy()


train_y = train_y.replace('Ia', 'event')
train_y = train_y.replace('Ib', 'event')
train_y = train_y.replace('II', 'event') 

clf = LogisticRegression(random_state=0)

#se on ain kivaa tehä jotai yleist funktioo ja tajuta myöhemmin et se ei toimi ku vaa yhelle mallile
""" tree = DecisionTreeClassifier(random_state=0) 
nb = GaussianNB() """

models = [clf]
model_names = ['LR']

coef_lists = []
for model in models:
  cv_results = cross_validate(model, train_x, train_y2, cv=5, scoring='accuracy', return_estimator=True)
  est_coefs = [est.coef_ for est in cv_results['estimator']]

  temp = np.array(est_coefs)

  #Multiclass tai binary
  temp = np.mean(np.abs(temp), axis=1)
  #temp = np.squeeze(temp)


  temp = pd.DataFrame(temp)
  temp = temp.agg(['mean'])
  coef_lists.append(temp.T['mean'].to_list())

coef_lists = np.array(coef_lists).T.tolist()
coef_df = pd.DataFrame(data=coef_lists, index=train_x.columns, columns=model_names)
coef_df = coef_df.sort_values('LR', key=abs, ascending=False)

print(coef_df.head(20))

# Multiclass
#LR 5 Best Coefficients: ['CS.mean', 'SO2168.mean', 'SWS.mean', 'Glob.mean', 'SWS.std']
#LR 5 wost Coefficients: ['T168.std', 'NOx504.mean','H2O336.std', 'O384.mean', 'O3168.mean']


# Binary
#LR 5 Best Coefficients: ['CS.mean', 'SWS.std', 'RHIRGA672.mean', 'PAR.mean', 'Glob.mean']
#LR 5 wost Coefficients: ['NOx672.mean', 'CO2504.std','NO168.std', 'NO336.mean', 'CO2504.std']



best5 = ['CS.mean', 'SWS.std', 'RHIRGA672.mean', 'PAR.mean', 'Glob.mean']
worst5 = ['NOx672.mean', 'CO2504.std','NO168.std', 'NO336.mean', 'CO2504.std']

best52 = ['CS.mean', 'SO2168.mean', 'SWS.mean', 'Glob.mean', 'SWS.std']
worst52 = ['T168.std', 'NOx504.mean','H2O336.std', 'O384.mean', 'O3168.mean']

fig, axs = plt.subplots(2,5)

nonevent_df = df[df['class4'] == 'nonevent']
event_df = df[df['class4'] != 'nonevent']
ia_df = df[df['class4'] == 'Ia']
ib_df = df[df['class4'] == 'Ib']
ii_df = df[df['class4'] == 'II']


""" for i in range(0,10):

  if i < 5:
    axs[0,i].set_title(best5[i])
    axs[0,i].boxplot([nonevent_df[best5[i]], event_df[best5[i]]], tick_labels=['non', 'event'], sym='_')
  else:
    axs[1,i-5].set_title(worst5[i-5])
    axs[1,i-5].boxplot([nonevent_df[worst5[i-5]], event_df[worst5[i-5]]], tick_labels=['non', 'event'], sym='_') """


for i in range(0,10):

  if i < 5:
    axs[0,i].set_title(best5[i])
    l = [nonevent_df[best5[i]],event_df[worst5[i-5]], ia_df[best5[i]], ib_df[best5[i]], ii_df[best5[i]]]
    axs[0,i].boxplot(l, tick_labels=['non','any', 'Ia', 'Ib', 'II'], sym='_')
  else:
    axs[1,i-5].set_title(worst5[i-5])
    l = [nonevent_df[worst5[i-5]],event_df[worst5[i-5]], ia_df[worst5[i-5]], ib_df[worst5[i-5]], ii_df[worst5[i-5]]]
    axs[1,i-5].boxplot(l, tick_labels=['non','any', 'Ia', 'Ib', 'II'], sym='_')

plt.tight_layout()
plt.show()