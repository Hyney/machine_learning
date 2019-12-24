from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from k_means import KMeans

iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_types = iris_data.target_names
data['class'] = iris_data.target
orm_dict = dict(zip(iris_types, [0, 1, 2]))
x_axis = 'petal length (cm)'
y_axis = 'petal width (cm)'
figure = plt.figure('KMeans', figsize=(10, 8))
ax = figure.add_subplot(221, title='label known')
for iris_type in iris_types:
    ax.scatter(
        data.loc[data['class'] == orm_dict[iris_type], x_axis],
        data.loc[data['class'] == orm_dict[iris_type], y_axis],
        label=iris_type,
        marker='.'
    )
ax2 = figure.add_subplot(222, title='label unknown')
ax2.scatter(data[x_axis], data[y_axis], marker='.')


num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
y_train = data['class'].values.reshape((num_examples, 1))

num_clusters = 3
max_iteration = 50

estimator = KMeans(x_train, num_clusters)
closest_centroids_ids, centroids = estimator.train(max_iteration)

ax3 = figure.add_subplot(223, title='KMeans')
for centroid_id, centroid in enumerate(centroids):
    current_examples_index = (closest_centroids_ids == centroid_id).flatten()
    ax3.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], label=centroid_id, marker='.')

for centroid_id, centroid in enumerate(centroids):
    ax3.scatter(centroid[0], centroid[1], c='black', marker='x')
figure.show()
ax.legend()
ax3.legend()
plt.show()
