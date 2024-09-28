import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone

mnist_data = pd.read_csv('../datasets/mnist_train.csv')
mnist_features = mnist_data[mnist_data.columns[1:]]
mnist_labels = mnist_data['label']

mnist_features = np.asarray(mnist_features)
mnist_features = mnist_features/255.

x_train, x_test, y_train, y_test = train_test_split(mnist_features, mnist_labels, shuffle=True, test_size=0.2)

logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')

from sklearn.model_selection import GridSearchCV
# logistic_param_grid = [{'C': [0.1, 1, 5]}]
# logistic_grid_search = GridSearchCV(logistic, logistic_param_grid, cv=2)
# logistic_grid_search.fit(x_train, y_train)

rbm = BernoulliRBM(verbose=True)
rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
rbm.learning_rate = 0.06
# logistic.C = logistic_grid_search.best_params_['C']

param_grid = [{'rbm__n_components': [5, 50, 100, 150], 'rbm__n_iter': [5, 20]}]
grid_search = GridSearchCV(rbm, param_grid, cv=2, scoring='accuracy')
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)

for i in range(8):
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean test score: ', grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])