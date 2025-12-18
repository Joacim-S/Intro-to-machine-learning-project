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

#tää on siis subset selectioo tärkeint täs on niinku mitkä ne featuret on jotka antaa parhaan tuloksen
#sen siis saa tol sel1.get_feature_names_out() sit ku se on fitattu

df = pd.read_csv('train.csv')

df = df.drop(columns=['id', 'date', 'partlybad'])

train_x = df.drop(columns=['class4'])
train_x = (train_x-train_x.mean())/train_x.std()
train_y = df['class4']
train_y2 = train_y.copy()
train_x2 = train_x.copy()

#2-tier
train_x = train_x[train_y != 'nonevent']
train_y = train_y[train_y != 'nonevent']

""" print(len(train_x),len(train_y),len(train_x2),len(train_y2) )
exit() """

columns = train_x.columns

train_y2 = train_y2.replace('Ia', 'event')
train_y2 = train_y2.replace('Ib', 'event')
train_y2 = train_y2.replace('II', 'event')
    

clf_lasso = LogisticRegression(random_state=0, solver='saga', penalty='l1', C=1, tol=0.001, max_iter=1000)
clf_ridge = LogisticRegression(random_state=0, solver='saga', penalty='l2', C=1, tol=0.001, max_iter=1000)
clf_DT = DecisionTreeClassifier(random_state=0)
clf_RF = RandomForestClassifier(random_state=0)
clf_SVC = SVC(kernel='linear',random_state=0)
#clf_NB = GaussianNB()

def subset_selection(x, y):

    sel1 = RFECV(clf_lasso, min_features_to_select=1, step=1)
    sel1.fit(x, y)

    sel2 = RFECV(clf_ridge, min_features_to_select=1, step=1)
    sel2.fit(x, y)

    sel3 = RFECV(clf_DT, min_features_to_select=1, step=1)
    sel3.fit(x, y)

    sel4 = RFECV(clf_RF, min_features_to_select=1, step=1)
    sel4.fit(x, y)

    sel5 = RFECV(clf_SVC, min_features_to_select=1, step=1)
    sel5.fit(x, y)


    #different outputs
    """ print('CV_results:', sel1.cv_results_)
    print('feature_names:', sel1.feature_names_in_)
    print('number of features:', sel1.n_features_)
    print('number of features during fit:', sel1.n_features_in_)
    print(sel1.ranking_) """


    selections = [sel1, sel2, sel3, sel4, sel5]

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
        opt_accuracy.append(data["mean_test_score"][sel.n_features_-1])
        avg_accuracy.append(data["mean_test_score"][99])

    model = ['LR L1', 'LR L2', 'DT', 'RF', 'SVC']

    RFE_df = pd.DataFrame({ 'Model': model, 'Optimal n of features': opt_features, 'Best accuracy': opt_accuracy, 'Accuracy without RFE': avg_accuracy})
    RFE_df.set_index('Model')

    return RFE_df, cv_results

RFE_df_binary, cv_results_binary = subset_selection(train_x2, train_y2)
RFE_df_multi, cv_results_multi = subset_selection(train_x, train_y)
np.save('binaryCV.npy',cv_results_binary)
np.save('multiCV.npy',cv_results_multi)
RFE_df_binary.to_csv('RFE_binary.csv')
RFE_df_multi.to_csv('RFE_multi.csv')

#cv_results_binary = np.array(cv_results_binary)
""" print(RFE_df_binary)
print(RFE_df_multi) """

""" LR1_results = cv_results_binary[0][:40]
LR1_results2 = cv_results_multi[0][:40]

LR2_results = cv_results_binary[1][:40]
LR2_results2 = cv_results_multi[1][:40]

DT_results2 = cv_results_multi[2][:40]
DT_results = cv_results_binary[2][:40]

RF_results2 = cv_results_multi[3][:40]
RF_results = cv_results_binary[3][:40]

SVC_results2 = cv_results_multi[4][:40]
SVC_results = cv_results_binary[4][:40]

plt.subplot(1,2,1)
plt.plot(LR1_results["n_features"], LR1_results["mean_test_score"], color='red', label='LR (Lasso)')
plt.plot(LR2_results["n_features"], LR2_results["mean_test_score"], color='blue', label='LR (Ridge)')
plt.plot(DT_results["n_features"], DT_results["mean_test_score"], color='green', label='Decicion Tree')
#plt.plot(DT_results["n_features"], DT_results["mean_test_score"], color='green', label='Decicion Tree')


plt.legend()
plt.title('RFE binary classification accuracy')

plt.subplot(1,2,2)
plt.plot(LR1_results2["n_features"], LR1_results2["mean_test_score"], color='red', label='LR (Lasso)')
plt.plot(LR2_results2["n_features"], LR2_results2["mean_test_score"], color='blue', label='LR (Ridge)')
plt.plot(DT_results2["n_features"], DT_results2["mean_test_score"], color='green', label='Decicion Tree')

plt.legend()
plt.title('RFE multiclass accuracy')

plt.tight_layout()
plt.show() """