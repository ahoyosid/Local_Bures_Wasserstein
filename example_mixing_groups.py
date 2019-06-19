import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegressionCV
from approximated_transport import LBWMixEstimator, MappingTransportEstimator


### Load wine dataset
data = pd.read_csv("data/Wine_Quality_Data.csv").fillna(0)

column_names = data.columns
y = data[column_names[column_names.isin(["quality>=6"])]].values.ravel().astype("int").ravel()
# Here, the last column is the sensitive feature (color: is white or red?)
X = data[column_names[~column_names.isin(["quality>=6"])]].values.astype("float")

# Sensitive group is Color (last column)
z = X[:, -1]

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, random_state=53)

### Preprocessing
scaler = StandardScaler()
pca = PCA(n_components=7, random_state=43)
pipeline = Pipeline(steps=[("scaler", scaler), ("pca", pca)]).fit(X_train)

X_train_ = pipeline.transform(X_train)
X_test_ = pipeline.transform(X_test)

X_train_torch = torch.tensor(X_train_).float()
X_test_torch = torch.tensor(X_test_).float()
z_train_torch = torch.tensor(z_train)
z_test_torch = torch.tensor(z_test)


### Mixing groups with L-BW
mix_estimator = LBWMixEstimator(n_components=1, random_state=43)

"""
kernel = "linear"
eta = 1e-3
sigma = 1
mu = 1
bias = True if kernel == "linear" else False
n_components = 100

mix_estimator = MappingTransportEstimator(
    n_components=n_components, 
    eta=eta, sigma=sigma, mu=mu, bias=bias,
    kernel=kernel, barycenter=None, 
    path="ME_%s_model" % kernel)
"""

mix_estimator.fit(X_train_torch, z_train_torch)

X_train_transported = mix_estimator.transform(
    X_train_torch, z_train_torch)

X_test_transported = mix_estimator.transform(
    X_test_torch, z_test_torch)

### Plot first two principal components
fig, axx = plt.subplots(1, 2, figsize=[10, 3])

axx[0].scatter(*X_train_[z_train == 1, :2].T, label="Group A")
axx[0].scatter(*X_train_[z_train != 1, :2].T, label="Group B")

axx[1].scatter(*X_train_transported[z_train == 1, :2].T, label="Group A")
axx[1].scatter(*X_train_transported[z_train != 1, :2].T, label="Group B")

for ax in axx:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"PC$_1$")
    ax.set_ylabel(r"PC$_2$")

axx[0].set_title("Input data")
axx[1].set_title("Mixing sensitive groups")

sns.despine()
plt.legend()
plt.savefig("principal_components_mixing_groups.png", bbox_inches='tight')


### Train a classifiers
clf_raw = LogisticRegressionCV(cv=5, penalty="l2", 
                               solver="lbfgs", random_state=45)
clf_fair = LogisticRegressionCV(cv=5, penalty="l2", 
                                solver="lbfgs", random_state=45)

clf_raw.fit(X_train_, y_train)
clf_fair.fit(X_train_transported, y_train)

y_pred_raw = clf_raw.predict_proba(X_test_)[:, 1]
y_pred_fair = clf_fair.predict_proba(X_test_transported)[:, 1]

### Display performance
def fairness_score(y_true, z_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    sfpr, stpr, _ = roc_curve(z_true, y_pred)

    return fpr, tpr, sfpr, stpr


fig = plt.figure(figsize=[5, 5])
fpr_raw, tpr_raw, sfpr_raw, stpr_raw = fairness_score(y_test, z_test, y_pred_raw)
fpr_fair, tpr_fair, sfpr_fair, stpr_fair = fairness_score(y_test, z_test, y_pred_fair)

diff_raw = stpr_raw - sfpr_raw 
diff_fair = sfpr_fair - stpr_fair
delta_raw = np.abs(np.max(diff_raw) - np.min(diff_raw))
delta_fair = np.abs(np.max(diff_fair) - np.min(diff_fair))

plt.plot(fpr_raw, tpr_raw, color="b", ls="--", label=r"y$_{pred}$ Raw")
plt.plot(sfpr_raw, stpr_raw, color="m", ls="--", label="Fairness Raw: %.2f" % delta_raw)

plt.plot(fpr_fair, tpr_fair, color="b", ls="-", label=r"y$_{pred}$ Fair")
plt.plot(sfpr_fair, stpr_fair, color="m", ls="-", label="Fairness Fair: %.2f" % delta_fair)

plt.plot([0, 1], [0, 1], color='k', label="Random choice")
legend = plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize=14, title="Prediction")
legend.get_title().set_fontsize(16)
plt.xlabel("False positive rate", fontsize=16)
plt.ylabel("True positive rate", fontsize=16)
plt.savefig("prediction_mixing_groups.png", bbox_inches='tight')