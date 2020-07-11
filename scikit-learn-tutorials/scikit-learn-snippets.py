import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame()                                   # dummy void data frame
features = []                                           # selected features columns
target_feature = ""                                     # feature we want to predict

X = data[features]
y = data[target_feature]



# R LIKE LINEAR REGRESSION SUMMARY
X2 = sm.add_constant(X)
ols = sm.OLS(y, X2)
ols_model = ols.fit()
print(ols_model.summary())
# this will print a summary of OLS Regression Results like in R
# this is usefull to see the p-values for each feature selected



# SPLIT THE DATA IN TRAIN AND TEST SET ------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



# LINEAR REGRESSION MODEL -------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)

# k-fold validation within a linear model
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=None)
kf.get_n_splits(X)  # returns the number of splits
results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index,], X.loc[test_index,]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results.append(r2_score(y_test, predictions))
r2_mean = np.mean(results)

# Leave one out validation within a linear model
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X.loc[train_index,], X.loc[test_index,]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results.append(r2_score(y_test, predictions))



# BASIC NEAURAL NETWORK (NN) ------------------------------------------------------------------------
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPRegressor(activation="relu", solver='adam', alpha=0.0001, batch_size='auto', learning_rate="constant", hidden_layer_sizes=(5,), warm_start=True)
model = clf.fit(X_train, y_train)
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)

# manual parameter selection
alphas = [0.0001, 0.001, 0.01, 0.1]
layers = [5, 10, 50, 100]
solvers = ["adam", "lbfgs"]
for alpha in alphas:
    for layer in layers:
        for solver in solvers:
            clf = MLPRegressor(activation="relu", solver=solver, alpha=alpha, batch_size='auto', learning_rate="constant", hidden_layer_sizes=(layer,), warm_start=True)
            model = clf.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)

# auto parameter selection
from sklearn.model_selection import GridSearchCV

parameters = {
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "hidden_layer_sizes" : [5, 10, 50, 100],
    "solver" : ("adam", "lbfgs"),
    "learning_rate" : ("constant", "adaptative")
}
nn = MLPRegressor(warm_start=True, max_iter=100000)
clf = GridSearchCV(nn, parameters, cv=5, n_jobs=-1) # cv=cross validation, n_jobs=parallelize
clf.fit(X,y)
clf.best_params_ # best parameters




# XGBoost ---------------------------------------------------------------------------------------
# https://xgboost.readthedocs.io/en/latest/
from xgboost import XGBRegressor

scaler = StandardScaler()
scaler.fit(X_train)
model = XGBRegressor(max_depth=None,learning_rate=None, n_estimators=100, verbosity=None, booster=None,
                    tree_method=None, n_jobs=None, gamma=None, min_child_weight=None, max_delta_step=None,
                    subsample=None, colsample_bytree=None, colsample_bylevel=None, colsample_bynode=None, reg_alpha=None,
                    reg_lambda=None, scale_pos_weight=None, base_score=None, random_state=None, missing=np.nan, num_parallel_tree=None,
                    monotone_constraints=None, interaction_constraints=None, importance_type="gain", gpu_id=None, validate_parameters=None)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)

# FEATURE SELECTION WITH RFE --------------------------------------------------------------------
# recursive feature elimination
from sklearn.feature_selection import RFE

model = LinearRegression()
model.fit(X_train, y_train)
selector = RFE(estimator=model, n_features_to_select=5, step=1, verbose=0)
selector.fit(X,Y)
X.columns[selector.support_]  # returns the selected columns by the RFE



# PCA ---------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/decomposition.html?highlight=decomposing%20signals%20components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
np.round(pca.components_)
pca.explained_variance_ratio_
new_X = pca.transform(X)
plt.scatter(new_X[:,0], new_X[:,1])



# DATA IMPUTATION --------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

# mean imputation
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp.fit(X)
X2 = imp.transform(X)
X2 = pd.DataFrame(X2)


# imputation by correlation
X.corr() # see the correlation between the desired columns
cols = ["cols that have correlation in with the column that you want to clean"]
X_new = X[cols] # also we need to drop na's from the column we want to clean
lr_model = LinearRegression()
lr_model.fit(X_new["feature cols"],X_new["imputation col"])
coef = lr_model.coef_

values_to_impute = X_new[np.isnan(X_new["imputation col"])]
values_to_impute = values_to_impute["cols that we used in the lr model"]
new_values = lr_model.predict(values_to_impute)

X.loc[isnan(X_new["imputation col"]), "imputation col"] = new_values

