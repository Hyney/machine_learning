#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author  :   Hyney
@Software:   PyCharm
@File    :   logicstic_case_with_linear_boundary.py
@Time    :   2021/5/2 20:01
@Desc    :   逻辑回归-多分类，鸢尾花数据集-线性决策边界
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from custom_logistic.logistic import LogisticRegression
from LogisticRegression.logistic_regression import LogisticRegression


def test_iris():
    # 1、加载鸢尾花数据集
    data = pd.read_csv('../data/iris.csv')
    iris_types = data['class'].unique()  # 花的类别
    features = [i for i in data.keys() if i != 'class']
    x_axis = 'petal_length'
    y_axis = 'petal_width'
    for iris_type in iris_types:
        plt.scatter(
            data[x_axis][data['class'] == iris_type],
            data[y_axis][data['class'] == iris_type],
            label=iris_type
        )
    plt.legend()
    plt.show()

    num_examples = data.shape[0]
    x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
    y_train = data['class'].values.reshape((num_examples, 1))
    estimator = LogisticRegression(x_train, y_train)
    cost = estimator.train()
    for index, label in enumerate(estimator.unique_labels):
        plt.plot(range(len(cost[index])), cost[index], label=label)
    plt.show()

    y_train_predictions = estimator.predict(x_train)

    precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
    print(precision)


if __name__ == '__main__':
    test_iris()