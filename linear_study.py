# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from LinearRegression import LinearRegression


def word_happy():
    data = pd.read_csv('./data/world-happiness-report-2017.csv')
    data_train = data.sample(frac=0.8)        # 获取训练集
    data_test = data.drop(data_train.index)   # drop掉训练集
    x_train = data_train[['Economy..GDP.per.Capita.']]
    y_train = data_train[['Happiness.Score']]

    x_test = data_test[['Economy..GDP.per.Capita.']]
    y_test = data_test[['Happiness.Score']]

    figure = plt.figure('test', figsize=[10, 6])
    ax = figure.add_subplot(221, title='linear')
    # ax = figure.add_axes([0.1, 0.1, 0.8, 0.8], label='test word happy')
    ax.scatter(x_train, y_train, color='r', marker='.', label='Train')
    ax.scatter(x_test, y_test, color='g', marker='v', label='Test')
    ax.set_xlabel('Economy..GDP.per.Capita.')
    ax.set_ylabel('Happiness.Score')

    ax2 = figure.add_subplot(222, title='loss')
    estimator = LinearRegression(x_train, y_train)
    loss = estimator.train(0.01, 500)
    ax2.plot(range(500), loss, label='loss')
    ax2.set_xlabel('iter')
    ax2.set_ylabel('loss_value')

    y_pre = estimator.predict(x_test)
    ax.plot(x_test, y_pre, color='b', label='line')
    ax.legend()
    figure.show()
    plt.show()


if __name__ == '__main__':
    word_happy()
