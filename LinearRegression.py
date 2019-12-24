# -*- coding: utf-8 -*-

import numpy as np

from machine_learning.utils.features import prepare_for_training


class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1、对数据进行预处理
        2、先得到所有的参数个数
        3、初始化参数矩阵
        :param data:
        :param labels:
        :param polynomial_degree:
        :param sinusoid_degree:
        :param normalize_data:
        """
        data_processed, features_mean, features_deviation = prepare_for_training(
            data, polynomial_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iteration=500):
        """

        :param alpha: 学习率
        :param num_iteration: 迭代次数
        :return:
        """
        cost_history = self.gradient_descent(alpha, num_iteration)
        return cost_history

    def gradient_descent(self, alpha, num_iteration):
        """
        梯度下降，迭代更新
        :param alpha: 学习率
        :param num_iteration: 迭代次数
        :return:
        """
        cost_loss = []
        for _ in range(num_iteration):
            self.gradient_step(alpha)
            cost_loss.append(self.cost(self.data, self.labels))
        return cost_loss

    def gradient_step(self, alpha):
        """
        梯度计算并更新
        :param alpha: 学习率
        :return:
        """
        y_pre = np.dot(self.data, self.theta)
        delta = y_pre - self.labels
        self.theta -= alpha * (np.dot(delta.T, self.data)).T / (self.data.shape[0])
        # theta(delta)对应的每个数据的特征，data一个多维矩阵，列为数据的特征，每一行为样本数据；因此delta需要进行转置为行矩阵(现为列矩阵)。常规下定义的theta因为列矩阵，所以计算后需将上一步的行矩阵转置为列矩阵

        # self.theta -= alpha * (delta * self.data).mean(axis=0)
        # return self.theta

    def cost(self, data, labels):
        """
        损失函数
        :param data:
        :param labels:
        :return:
        """
        y_pre = np.dot(self.data, self.theta)
        delta = y_pre - labels
        loss_data = np.dot(delta.T, delta) * (1/2) / data.shape[0]
        return loss_data[0][0]

    def loss(self, data):
        """
        损失值
        :param data:
        :return:
        """
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        return data_processed

    def predict(self, data):
        """
        预测
        :param data:
        :return:
        """
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        return np.dot(data_processed, self.theta)
