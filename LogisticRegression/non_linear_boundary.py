
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from LogisticRegression.logistic_regression import LogisticRegression


data = pd.read_csv('../data/microchips-tests.csv')
label = data.columns[-1]
label_datas = np.unique(data[label].values)
x_axis = data.columns[0]
y_axis = data.columns[1]

for label_data in label_datas:
    plt.scatter(
        data[x_axis][data[label] == label_data],
        data[y_axis][data[label] == label_data],
        label=label_data
    )
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title('Microchips Tests')
plt.legend()
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
y_train = data[label].values.reshape((num_examples, 1))

max_iteration = 100000
regularization_parm = 0
polynomial_degree = 5
sinusoid_degree = 0

estimator = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
cost = estimator.train(max_iteration)
labels = estimator.unique_labels
plt.plot(range(len(cost[0])), cost[0], label=labels[0])
plt.plot(range(len(cost[1])), cost[1], label=labels[1])
plt.legend()
plt.show()
y_train_predictions = estimator.predict(x_train)

precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
print(precision)

x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])

X = np.linspace(x_min, x_max, num_examples)
Y = np.linspace(y_min, y_max, num_examples)

Z = np.zeros((num_examples, num_examples))

for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        predictions = estimator.predict(data)
        Z[x_index][y_index] = estimator.predict(data)[0, 0]

positives = (y_train == 1).flatten()
negatives = (y_train == 0).flatten()
plt.scatter(x_train[negatives, 0], x_train[negatives, 1], label='0')
plt.scatter(x_train[positives, 0], x_train[positives, 1], label='1')
plt.contour(X, Y, Z)
plt.show()
