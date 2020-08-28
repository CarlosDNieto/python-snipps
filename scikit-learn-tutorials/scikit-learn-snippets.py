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



# LABEL ENCODER ------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

# print("MAE from Approach 2 (Label Encoding):") 
# print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))

# print('Categorical columns that will be label encoded:', good_label_cols)
# print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

# print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
# print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)


# ONE HOT ENCODER ------------------------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# print("MAE from Approach 3 (One-Hot Encoding):") 
# print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))