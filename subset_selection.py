import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

#tää on siis subset selectioo tärkeint täs on niinku mitkä ne featuret on jotka antaa parhaan tuloksen
#sen siis saa tol sel1.get_feature_names_out() sit ku se on fitattu

df = pd.read_csv('train.csv')

df = df.drop(columns=['id', 'date', 'partlybad'])
#df = df.iloc[:50]

train_x = df.drop(columns=['class4'])
train_x = (train_x-train_x.mean())/train_x.std()
train_y = df['class4']
train_y2 = train_y.copy()

columns = train_x.columns

train_y2 = train_y2.replace('Ia', 'event')
train_y2 = train_y2.replace('Ib', 'event')
train_y2 = train_y2.replace('II', 'event')
    
def nb_ig(nb, x, y):
    res = permutation_importance(nb, x, y, n_repeats=5, random_state=0, scoring='accuracy')
    return res.importances_mean

clf_lasso = LogisticRegression(random_state=0, solver='saga', penalty='l1', C=1, tol=0.001, max_iter=1000)
clf_ridge = LogisticRegression(random_state=0, solver='saga', penalty='l2', C=1, tol=0.001, max_iter=1000)
clf_DT = DecisionTreeClassifier()
#clf_NB = GaussianNB()

sel1 = RFECV(clf_lasso, min_features_to_select=1, step=1)
sel1.fit(train_x, train_y2)

sel2 = RFECV(clf_ridge, min_features_to_select=1, step=1)
sel2.fit(train_x, train_y2)

sel4 = RFECV(clf_DT, min_features_to_select=1, step=1)
sel4.fit(train_x, train_y2)

""" sel3 = RFECV(clf_NB, min_features_to_select=1, step=1, importance_getter=nb_ig)
sel3.fit(train_x, train_y2) """


#different outputs
""" print('CV_results:', sel1.cv_results_)
print('feature_names:', sel1.feature_names_in_)
print('number of features:', sel1.n_features_)
print('number of features during fit:', sel1.n_features_in_)
print(sel1.ranking_) """


selections = [sel1, sel2, sel4]

cv_results = []
opt_features = []
opt_accuracy = []
avg_accuracy = []

for sel in selections:
    data = {
        key: value
        for key, value in sel.cv_results_.items()
        if key in ["n_features", "mean_test_score", "std_test_score"]
    }
    cv_results.append(pd.DataFrame(data))
    opt_features.append(sel.n_features_)
    opt_accuracy.append(data["mean_test_score"][sel.n_features_])
    avg_accuracy.append(data["mean_test_score"][99])

model = ['LR L1', 'LR L2', 'DT']

RFE_df = pd.DataFrame({ 'Model': model, 'Optimal n of features': opt_features, 'Best accuracy': opt_accuracy, 'Accuracy without RFE': avg_accuracy})
RFE_df.set_index('Model')


""" print(RFE_df)

LR1_results = cv_results[0].iloc[:40]
LR2_results = cv_results[1].iloc[:40]
DT_results = cv_results[2].iloc[:40]
#NB_results = cv_results[2].iloc[:40]

plt.plot(LR1_results["n_features"], LR1_results["mean_test_score"], color='red', label='LR (Lasso)')
plt.plot(LR2_results["n_features"], LR2_results["mean_test_score"], color='blue', label='LR (Ridge)')
plt.plot(DT_results["n_features"], DT_results["mean_test_score"], color='green', label='Decicion Tree')
#plt.plot(NB_results["n_features"], NB_results["mean_test_score"], color='green', label='Naive Bayes')

plt.legend()
plt.title('RFE binary classification accuracy')
plt.show() """