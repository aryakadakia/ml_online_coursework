import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, Lars, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings("ignore")

automobile_df = pd.read_csv('../datasets/auto_mpg_processed.csv')

results_dict = {}


def build_model(regression_fn, name_of_y_col, names_of_x_cols, dataset, test_frac=0.2, preprocess_fn=None,
                show_plot_Y=False, show_plot_scatter=False):
    X = dataset[names_of_x_cols]
    Y = dataset[name_of_y_col]

    if preprocess_fn is not None:
        X = preprocess_fn(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    model = regression_fn(x_train, y_train)
    y_pred = model.predict(x_test)

    if show_plot_Y == True:
        # fig, ax = plt.subplots(figsize=(12, 8))
        # plt.plot(y_pred, label="Predicted")
        # plt.plot(y_test.values, label="Actual")
        # plt.ylabel(name_of_y_col)
        print()
        # plt.legend()
        # plt.show()

    if show_plot_scatter == True:
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.scatter(x_test, y_test)
        plt.plot(x_test, y_pred, 'r')
        plt.legend(['Predicted line', 'Observed data'])
        plt.show()

    return {'training_score': model.score(x_train, y_train), 'test_score': r2_score(y_test, y_pred)}


def compare_results():
    for key in results_dict:
        print('Regression: ', key)
        print('Training score: ', results_dict[key]['training_score'])
        print('Test score: ', results_dict[key]['test_score'])
        print()


def linear_reg(x_train, y_train):
    model = LinearRegression(normalize=True)
    model.fit(x_train, y_train)
    return model


results_dict['mpg ~ single_linear'] = build_model(linear_reg, 'mpg', ['weight'], automobile_df, show_plot_Y=True)

results_dict['mpg ~ kitchen_sink_linear'] = build_model(linear_reg, 'mpg', ['cylinders', 'displacement', 'horsepower',
                                                                            'weight', 'acceleration'], automobile_df,
                                                        show_plot_Y=True)

results_dict['mpg ~ parsimonious_linear'] = build_model(linear_reg, 'mpg', ['horsepower', 'weight'], automobile_df,
                                                        show_plot_Y=True)


# l1 regularization
def lasso_reg(x_train, y_train, alpha=0.5):
    model = Lasso(alpha=alpha)
    model.fit(x_train, y_train)
    return model


# default alpha is 1 (what is multiplied to the l1 regularization). If set to 0, just regular linear regression


results_dict['mpg ~ kitchen_sink_lasso'] = build_model(lasso_reg, 'mpg', ['cylinders', 'displacement', 'horsepower',
                                                                          'weight', 'acceleration'], automobile_df,
                                                       show_plot_Y=True)


# l2 regularization
def ridge_reg(x_train, y_train, alpha=0.5, normalize=True):
    model = Ridge(alpha=alpha, normalize=normalize)
    model.fit(x_train, y_train)
    return model


results_dict['mpg ~ kitchen_sink_ridge'] = build_model(ridge_reg, 'mpg', ['cylinders', 'displacement', 'horsepower',
                                                                          'weight', 'acceleration'], automobile_df,
                                                       show_plot_Y=True)


# applies both l1 and l2 regularization
def elastic_net_reg(x_train, y_train, alpha=1, l1_ratio=0.5, normalize=False, max_iter=100000, warm_start=True,
                    equivalent_to="Elastic Net"):
    print("Equivalent to:", equivalent_to)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, normalize=normalize, max_iter=max_iter, warm_start=warm_start)
    model.fit(x_train, y_train)
    return model


# 0 = l1_ratio is just l2 regularization, 1 is l2
# warm start is reusing the solution of the previous call to fit as initialization

from functools import partial

# ordinary least squares
results_dict['mpg ~ kitchen_sink_elastic_net_ols'] = build_model(partial(elastic_net_reg, alpha=0, equivalent_to="OLS"),
                                                                 'mpg', ['cylinders', 'displacement', 'horsepower',
                                                                         'weight', 'acceleration'], automobile_df,
                                                                 show_plot_Y=True)

results_dict['mpg ~ kitchen_sink_elastic_net_lasso'] = build_model(partial(elastic_net_reg, alpha=1, l1_ratio=1,
                                                                           equivalent_to="Lasso"), 'mpg', ['cylinders',
                                                                                                           'displacement',
                                                                                                           'horsepower',
                                                                                                           'weight',
                                                                                                           'acceleration'],
                                                                   automobile_df, show_plot_Y=True)

results_dict['mpg ~ kitchen_sink_elastic_net_ridge'] = build_model(partial(elastic_net_reg, alpha=1, l1_ratio=0,
                                                                           equivalent_to="Ridge"), 'mpg', ['cylinders',
                                                                                                           'displacement',
                                                                                                           'horsepower',
                                                                                                           'weight',
                                                                                                           'acceleration'],
                                                                   automobile_df, show_plot_Y=True)

results_dict['mpg ~ kitchen_sink_elastic_net'] = build_model(partial(elastic_net_reg, alpha=1, l1_ratio=0.5, ), 'mpg',
                                                             ['cylinders', 'displacement', 'horsepower', 'weight',
                                                              'acceleration'], automobile_df, show_plot_Y=True)


# support vector regression, linear SVR
def svr_reg(x_train, y_train, kernel='linear', epsilon=0.05, C=0.3):
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(x_train, y_train)
    return model


# epsilon determines margin, C determines error violation (higher is greater penalty)

results_dict['mpg ~ kitchen_sink_svr'] = build_model(svr_reg, 'mpg', ['cylinders', 'displacement', 'horsepower',
                                                                      'weight', 'acceleration'], automobile_df,
                                                     show_plot_Y=True)


# k-nearest neighbors
def kneighbors_reg(x_train, y_train, n_neighbors=10):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    return model


results_dict['mpg ~ kitchen_sink_kneighbors'] = build_model(kneighbors_reg, 'mpg', ['cylinders', 'displacement',
                                                                                    'horsepower', 'weight',
                                                                                    'acceleration'], automobile_df,
                                                            show_plot_Y=True)


# standardize data
def apply_standardscaler(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler.transform(x)


# stochastic gradient regressor
def sgd_reg(x_train, y_train, max_iter=10000, tol=1e-3):
    model = SGDRegressor(max_iter=max_iter, tol=tol)
    model.fit(x_train, y_train)
    return model


results_dict['mpg ~ kitchen_sink_sgd'] = build_model(sgd_reg, 'mpg', ['cylinders', 'displacement',
                                                                      'horsepower', 'weight',
                                                                      'acceleration'], automobile_df,
                                                     show_plot_Y=True)


# decision tree
def decision_tree_reg(x_train, y_train, max_depth=2):
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(x_train, y_train)
    return model


results_dict['mpg ~ kitchen_sink_decision_tree'] = build_model(decision_tree_reg, 'mpg', ['cylinders', 'displacement',
                                                                                          'horsepower', 'weight',
                                                                                          'acceleration'],
                                                               automobile_df,
                                                               show_plot_Y=True)


# lars regression
def lars_reg(x_train, y_train, n_nonzero_coefs=4):
    model = Lars(n_nonzero_coefs=n_nonzero_coefs)
    model.fit(x_train, y_train)
    return model


results_dict['mpg ~ kitchen_sink_lars'] = build_model(lars_reg, 'mpg', ['cylinders', 'displacement',
                                                                        'horsepower', 'weight',
                                                                        'acceleration'],
                                                      automobile_df,
                                                      show_plot_Y=True)
# compare results
compare_results()
