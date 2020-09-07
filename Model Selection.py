import os
import sys
import numpy as np
import pandas as pd
from io import StringIO
from sklearn import metrics, svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

# Acknowledgement
# We have used the following resource in some codes
# www.ritchieng.com/machine-learning-efficiently-search-tuning-param


# cmd command
cmd = [i for i in sys.argv[1:]]  # we start from 1 since we just get desire arguments not the file name
A3_training_dataset = cmd[0]
A3_test_dataset = cmd[1]


number_of_splits = 5


# Loading dataset and create input, unseen and y DataFrames
def load_data():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    input_data = pd.read_table(A3_training_dataset)
    y = input_data.group
    input_data = input_data.drop(columns=['group'])
    unseen_data = pd.read_table(A3_test_dataset)
    return input_data, y, unseen_data


def knn(input_data, y):
    param_grid = dict(n_neighbors=range(1, 100), weights=['uniform', 'distance'])
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=number_of_splits, scoring='average_precision')
    grid.fit(input_data, y)
    grid_result = pd.DataFrame(grid.cv_results_)
    grid_result = grid_result.loc[:, 'params':'std_test_score']
    #print("KNN best score: ", grid.best_score_)
    #print("KNN best Param: ", grid.best_params_)
    #print("KNN result: ", grid_result)


def lin_regression(input_data, y):
    lin_reg = LinearRegression()
    param_grid = dict(fit_intercept=['True', 'False'], normalize=['True', 'False'])
    grid = GridSearchCV(lin_reg, param_grid, cv=number_of_splits, scoring='average_precision')
    grid.fit(input_data, y)
    grid_result = pd.DataFrame(grid.cv_results_)
    grid_result = grid_result.loc[:, 'params':'std_test_score']
    #print("lin_reg best score: ", grid.best_score_)
    #print("lin_reg best param: ", grid.best_params_)
    #print("lin_reg result: ", grid_result)


def log_regression(input_data, y):
    log_reg = LogisticRegression(max_iter=10000)
    param_grid = dict(fit_intercept=['True', 'False'], penalty=['l2', 'none'], solver=['newton-cg', 'sag'], tol=[1, 0.1, 0.01, 0.001])
    grid = GridSearchCV(log_reg, param_grid, cv=number_of_splits, scoring='average_precision')
    grid.fit(input_data, y)
    grid_result = pd.DataFrame(grid.cv_results_)
    grid_result = grid_result.loc[:, 'params':'std_test_score']
    #print(grid.classes_)  # this shows that the second element of each array is probability of being class 1
    #print("log_reg best score: ", grid.best_score_)
    #print("log_reg best param: ", grid.best_params_)
    #print("log_reg result: ", grid_result)


def random_forest(input_data, y):
    rand_forrest = RandomForestClassifier()
    param_grid = dict(n_estimators=range(10, 500), max_features=['sqrt', 'log2'], ccp_alpha=[1, 0.1, 0.01, 0.001])
    grid = GridSearchCV(rand_forrest, param_grid, cv=number_of_splits, scoring='average_precision')
    grid.fit(input_data, y)
    grid_result = pd.DataFrame(grid.cv_results_)
    grid_result = grid_result.loc[:, 'params':'std_test_score']
    #print('random_for best score: ', grid.best_score_)
    #print('random_for best param: ', grid.best_params_)
    #print("random_for result: ", grid_result)


def support_v_m(input_data, y):
    s_v_m = svm.SVC()
    param_grid_lin = dict(C=range(1, 100), kernel=['linear'])
    param_grid_pol = dict(C=[0.01, 0.1, 1, 10, 100], gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10], degree=[2], coef0=range(0, 100), kernel=['poly'])
    param_grid_rbf = dict(C=[0.01, 0.1, 1, 10, 100], gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10], kernel=['rbf'])
    param_grid_sig = dict(C=[0.01, 0.1, 1, 10, 100], gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10], coef0=range(0, 100), kernel=['sigmoid'])
    grid_lin = GridSearchCV(s_v_m, param_grid_lin, cv=number_of_splits, scoring='average_precision')
    grid_pol = GridSearchCV(s_v_m, param_grid_pol, cv=number_of_splits, scoring='average_precision')
    grid_rbf = GridSearchCV(s_v_m, param_grid_rbf, cv=number_of_splits, scoring='average_precision')
    grid_sig = GridSearchCV(s_v_m, param_grid_sig, cv=number_of_splits, scoring='average_precision')
    grid_lin.fit(input_data, y)
    grid_result = pd.DataFrame(grid_lin.cv_results_)
    #print("SVM_lin best score: ", grid_lin.best_score_)
    #print("SVM_lin best param: ", grid_lin.best_params_)
    #print("SVM_lin result: ", grid_result)
    grid_pol.fit(input_data, y)
    grid_result = pd.DataFrame(grid_pol.cv_results_)
    #print("SVM_pol best score: ", grid_pol.best_score_)
    #print("SVM_pol best param: ", grid_pol.best_params_)
    #print("SVM_pol result: ", grid_result)
    grid_rbf.fit(input_data, y)
    grid_result = pd.DataFrame(grid_rbf.cv_results_)
    #print("SVM_rbf best score: ", grid_rbf.best_score_)
    #print("SVM_rbf best param: ", grid_rbf.best_params_)
    #print("SVM_rbf result: ", grid_result)
    grid_sig.fit(input_data, y)
    grid_result = pd.DataFrame(grid_sig.cv_results_)
    #print("sigmoid best score: ", grid_sig.best_score_)
    #print("sigmoid best param: ", grid_sig.best_params_)
    #print("SVM_sig result: ", grid_result)


def plot():
    lables = ['KNN', 'LinReg', 'LogReg', 'RandomForest', 'SVM']
    values = [[0.805513, 0.836726, 0.830231, 0.803057, 0.810343],
              [0.855678, 0.808918, 0.82115, 0.843337, 0.835621],
              [0.823657, 0.84895, 0.846357, 0.826864, 0.839047],
              [0.81174, 0.827255, 0.819569, 0.795391, 0.813115],
              [0.819822, 0.847098, 0.839486, 0.820656, 0.828955]]
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))  # Building a figure
    axes.set_title('Best Models of Methods')
    bplot = axes.boxplot(values, labels=lables, vert=True, patch_artist=True)
    colors = 'teal'  # Coloring and Labeling
    for box in bplot['boxes']:
        box.set_facecolor(colors)
    axes.yaxis.grid(False)
    axes.set_xlabel('Methods')
    axes.set_ylabel('average_ precision')
    plt.savefig("A3")
    #plt.show()


def output():
    output_f = open("g08_predictions.txt", "w")
    output_f = open("g08_predictions.txt", "a")
    log_reg = LogisticRegression(penalty='none', solver='newton-cg', tol=0.002, max_iter=10000)
    log_reg.fit(input_data, y)
    prediction = log_reg.predict_proba(unseen_data)
    #print(log_reg.classes_)  # This shows the second element of each pair is the prediction of class 1
    prediction = [x[1] for x in prediction]
    for x in prediction:
        output_f.write(f'{x}\n')


# Normalizing input data
# Since normalize_input_data did not improve the performance, we ignored it.
# www.chrisalbon.com/python/data_wrangling/pandas_normalize_column
def normalize_input_data(input_data):
    col_names = list(input_data.columns)
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data)
    input_data = pd.DataFrame(input_data, columns=col_names)
    return input_data


# Lasso Feature Elimination
# Since the feature_selection_2 did not improve the performance, we ignored it.
# www.towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
def feature_selection_1(input_data, y):
    reg = LassoCV()
    reg.fit(input_data, y)
    #print("Best alpha using built-in LassoCV: ", reg.alpha_)
    #print("Best score using built-in LassoCV: ", reg.score(input_data, y))
    coef = pd.DataFrame(reg.coef_, index=input_data.columns)
    coef = np.array(reg.coef_)
    features = np.array(input_data.columns)
    feature_sel = []
    for i, x in enumerate(coef):
        if x == 0:
            feature_sel.append(features[i])
    input_data = input_data.drop(columns=feature_sel)
    return input_data


# Recursive Feature Elimination
# Since the feature_selection_2 did not improve the performance, we ignored it.
# www.machinelearningmastery.com/feature-selection-machine-learning-python/
def feature_selection_2(input_data, y):
    from sklearn.feature_selection import RFE
    model = LinearRegression()
    rfe = RFE(model, 73)
    fit = rfe.fit(input_data, y)
    features = np.array(input_data.columns)
    features_temp = fit.support_
    feature_drop = []
    for i, x in enumerate(features_temp):
        if x == False:
            feature_drop.append(features[i])
    input_data = input_data.drop(columns=feature_drop)
    return input_data
    #print(feature_drop)
    #print(f"Num Features: {fit.n_features_}")
    #print(f"Selected Features: {fit.support_}")
    #print(f"Feature Ranking: {fit.ranking_}")


input_data, y, unseen_data = load_data()
knn(input_data, y)
lin_regression(input_data, y)
log_regression(input_data, y)
random_forest(input_data, y)
support_v_m(input_data, y)
output()
#plot()





