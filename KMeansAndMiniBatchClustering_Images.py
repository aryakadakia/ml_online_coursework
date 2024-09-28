import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

mnist_data = pd.read_csv('../datasets/mnist_train.csv')

print(sorted(mnist_data['label'].unique()))

mnist_features = mnist_data.drop('label', axis=1)
mnist_labels = mnist_data['label']


def display_image(index):
    print("Digit: ", mnist_labels[index])
    plt.imshow(mnist_features.loc[index].values.reshape(28, 28), cmap='Greys')
    plt.show()


# display_image(5)

# model
kmeans_model = KMeans(max_iter=1000, n_clusters=10).fit(mnist_features)
kmeans_centroids = kmeans_model.cluster_centers_

# visualize centroids
# fig, ax = plt.subplots(figsize=(12, 8))
# for centroid in range(len(kmeans_centroids)):
#    plt.subplot(2, 5, centroid + 1)
#    plt.imshow(kmeans_centroids[centroid].reshape(28, 28), cmap='Greys')
#   plt.show()

# confirm unique clusters
print(np.unique(kmeans_model.labels_))

# creating test features and labels
mnist_test = mnist_data.sample(10, replace=False)
mnist_test_features = mnist_test.drop('label', axis=1)
mnist_test_labels = mnist_test['label']
mnist_test_labels = np.array(mnist_test_labels)

# predicted clusters using model on test data
pred_clusters = kmeans_model.predict(mnist_test_features)

pred_results = pd.DataFrame({'actual_digit': mnist_test_labels, 'pred_cluster': pred_clusters})
print(pred_results)

# do the same for minibatch
minibatch_kmeans_model = MiniBatchKMeans(n_clusters=10, max_iter=10000, batch_size=100).fit(mnist_features)