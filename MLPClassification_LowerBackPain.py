import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

spine_data = pd.read_csv('../datasets/Dataset_spine.csv', skiprows=1,
                         names=['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope',
                                'pelvic_radius', 'degree_spondylolisthesis', 'pelvic_slope', 'direct_tilt',
                                'thoracic_slope', 'cervical_tilt', 'sacrum_angle', 'scoliosis_slope', 'class'])

spine_data = spine_data.sample(frac=1).reset_index(drop=True)

from sklearn import preprocessing
label_encoding = preprocessing.LabelEncoder()
spine_data['class'] = label_encoding.fit_transform(spine_data['class'].astype(str))

spine_data_corr = spine_data.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(spine_data_corr, annot=True)
# plt.show()

from sklearn.model_selection import train_test_split
X = spine_data.drop('class', axis=1)
Y = spine_data['class']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, activation='logistic', alpha=0.001, solver='lbfgs',
                        verbose=True)

mlp_clf.fit(x_train, y_train)
y_pred = mlp_clf.predict(x_test)
pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print(pred_results.sample(5))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(y_test, y_pred))

# see how many values model got right and wrong
spine_data_crosstab = pd.crosstab(pred_results.y_test, pred_results.y_pred)
print(spine_data_crosstab)

print(confusion_matrix(y_test, y_pred))

# get precision, recall, and F1 score
print(classification_report(y_test, y_pred))