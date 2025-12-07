import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#tää on siis subset selectioo tärkeint täs on niinku mitkä ne featuret on jotka antaa parhaan tuloksen
#sen siis saa tol sel1.get_feature_names_out() sit ku se on fitattu

df = pd.read_csv('train.csv')

df = df.drop(columns=['id', 'date', 'partlybad'])

train_x = df.drop(columns=['class4'])
train_x = (train_x-train_x.mean())/train_x.std()
train_y = df['class4']
columns = train_x.columns

train_y = train_y.replace('Ia', 'event')
train_y = train_y.replace('Ib', 'event')
train_y = train_y.replace('II', 'event')
    


clf_lasso = LogisticRegression(random_state=0, solver='liblinear', penalty='l1').fit(train_x, train_y)
clf_ridge = LogisticRegression(random_state=0, solver='liblinear', penalty='l2').fit(train_x, train_y)

sel1 = RFECV(clf_lasso, min_features_to_select=1, step=1 )
sel1.fit(train_x, train_y)

sel2 = RFECV(clf_ridge, min_features_to_select=1, step=1 )
sel2.fit(train_x, train_y)

#different outputs
""" print('CV_results:', sel1.cv_results_)
print('feature_names:', sel1.feature_names_in_)
print('number of features:', sel1.n_features_)
print('number of features during fit:', sel1.n_features_in_)
print(sel1.ranking_) """

# 0 = LR with L1
# 1 = LR with L2
cv_results = []

data = {
    key: value
    for key, value in sel1.cv_results_.items()
    if key in ["n_features", "mean_test_score", "std_test_score"]
}
cv_results.append(pd.DataFrame(data))

data = {
    key: value
    for key, value in sel2.cv_results_.items()
    if key in ["n_features", "mean_test_score", "std_test_score"]
}
cv_results.append(pd.DataFrame(data))

model = ['LR L1', 'LR L2']
opt_features = [sel1.n_features_, sel2.n_features_]
opt_accuracy = [cv_results[0]["mean_test_score"][sel1.n_features_], cv_results[1]["mean_test_score"][sel2.n_features_]]
avg_accuracy = [cv_results[0]["mean_test_score"][99], cv_results[1]["mean_test_score"][99]]

RFE_df = pd.DataFrame({ 'Model': model, 'Optimal n of features': opt_features, 'Best accuracy': opt_accuracy, 'Accuracy without RFE': avg_accuracy})
#print(RFE_df)


""" print("Optimal number of features:")
print(f"For Lasso: {sel1.n_features_}, with accuracy of {cv_results[0]["mean_test_score"][sel1.n_features_]} (with all: {cv_results[0]["mean_test_score"][99]})")
print(f"For Ridge regression: {sel2.n_features_}, with accuracy of {cv_results[1]["mean_test_score"][sel2.n_features_]} (with all: {cv_results[1]["mean_test_score"][99]}")
 """
""" cv_results1 = cv_results[0].iloc[:40]
cv_results2 = cv_results[1].iloc[:40]

plt.suptitle("Recursive Feature Elimination for first 40 features")
plt.subplot(2, 1, 1)
plt.title("Lasso")
plt.ylabel("Accuracy")
plt.scatter(cv_results1["n_features"],cv_results1["mean_test_score"])
plt.plot(cv_results1["n_features"], cv_results1["mean_test_score"])

plt.subplot(2, 1, 2)
plt.title("Ridge Regression")
plt.xlabel("Features")
plt.scatter(cv_results2["n_features"],cv_results2["mean_test_score"])
plt.plot(cv_results2["n_features"], cv_results2["mean_test_score"])

plt.tight_layout()
plt.show() """

""" LR1_results = cv_results[0].iloc[:40]
LR2_results = cv_results[1].iloc[:40]

plt.plot(LR1_results["n_features"], LR1_results["mean_test_score"], color='red')
plt.plot(LR2_results["n_features"], LR2_results["mean_test_score"], color='blue')

plt.legend()
plt.title('RFE accuracy')
plt.show() """