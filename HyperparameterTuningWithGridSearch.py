import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

titanic_df = pd.read_csv('../datasets/titanic_processed.csv')

X = titanic_df.drop('Survived', axis=1)
Y = titanic_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print("Test data count: ", len(y_test))
    print("accuracy count: ", num_acc)
    print("accuracy score: ", acc)
    print("precision score: ", prec)
    print("recall score: ", rec)
    print()


tree_parameters = {'max_depth': [2, 4, 5, 7, 9, 10]}  # tuning the max depth hyperparameter
tree_grid_search = GridSearchCV(DecisionTreeClassifier(), tree_parameters, cv=3, return_train_score=True)
tree_grid_search.fit(x_train, y_train)
print(tree_grid_search.best_params_)
# cv = 3 means 3-fold cross validation (split dataset into 3 parts, 2 to train, 1 to evaluate)
# default scoring for classification is accuracy

for i in range(6):
    print("Parameters: ", tree_grid_search.cv_results_['params'][i])
    print("Meant Test Score: ", tree_grid_search.cv_results_['mean_test_score'][i])
    print("Rank: ", tree_grid_search.cv_results_['rank_test_score'][i])

decision_tree_model = DecisionTreeClassifier(max_depth=tree_grid_search.best_params_['max_depth']).fit(x_train, y_train)

tree_y_pred = decision_tree_model.predict(x_test)
summarize_classification(y_test, tree_y_pred)

logistic_parameters = {'penalty': ['l1', 'l2'], 'C': [0.1, 0.4, 0.8, 1, 2, 5]}
logistic_grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), logistic_parameters, cv=3,
                                    return_train_score=True)
logistic_grid_search.fit(x_train, y_train)

print(logistic_grid_search.best_params_)

for i in range(6):
    print("Parameters: ", logistic_grid_search.cv_results_['params'][i])
    print("Meant Test Score: ", logistic_grid_search.cv_results_['mean_test_score'][i])
    print("Rank: ", logistic_grid_search.cv_results_['rank_test_score'][i])

logistic_model = LogisticRegression(solver='liblinear',
                                    penalty=logistic_grid_search.best_params_['penalty'],
                                    C=logistic_grid_search.best_params_['C']).fit(x_train, y_train)
logistic_y_pred = logistic_model.predict(x_test)
summarize_classification(y_test, logistic_y_pred)

