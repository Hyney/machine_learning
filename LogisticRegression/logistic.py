from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from LogisticRegression.logistic_regression import LogisticRegression


iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_types = iris_data.target_names
data['class'] = iris_data.target
orm_dict = dict(zip(iris_types, [0, 1, 2]))
x_axis = 'petal length (cm)'
y_axis = 'petal width (cm)'
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == orm_dict[iris_type]],
                data[y_axis][data['class'] == orm_dict[iris_type]],
                label=iris_type,
                )
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
y_train = data['class'].values.reshape((num_examples, 1))

max_iteration = 10000
polynomial_degree = 0
sinusoid_degree = 0
estimator = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
cost = estimator.train(max_iteration)
labels = estimator.unique_labels
plt.plot(range(len(cost[0])), cost[0], label=labels[0])
plt.plot(range(len(cost[1])), cost[1], label=labels[1])
plt.plot(range(len(cost[2])), cost[2], label=labels[2])
plt.legend()
plt.show()

y_train_prediction = estimator.predict(x_train)
precision = np.sum(y_train_prediction == y_train) / y_train.shape[0] * 100
print(precision)

x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])

X = np.linspace(x_min, x_max, num_examples)
Y = np.linspace(y_min, y_max, num_examples)

Z_1 = np.zeros((num_examples, num_examples))
Z_2 = np.zeros((num_examples, num_examples))
Z_3 = np.zeros((num_examples, num_examples))
for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        prediction = estimator.predict(data)[0, 0]
        if prediction == 0:
            Z_1[x_index][y_index] = 1
        elif prediction == 1:
            Z_2[x_index][y_index] = 1
        elif prediction == 2:
            Z_3[x_index][y_index] = 1

for iris_type in iris_types:
    plt.scatter(
        x_train[(y_train == orm_dict[iris_type]).flatten(), 0],
        x_train[(y_train == orm_dict[iris_type]).flatten(), 1],
        label=iris_type
                )
plt.contour(X, Y, Z_1)
plt.contour(X, Y, Z_2)
plt.contour(X, Y, Z_3)
plt.show()
