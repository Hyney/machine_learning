#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author  :   Hyney
@Software:   PyCharm
@File    :   logistic.py
@Time    :   2021/4/26 20:28
@Desc    :
"""
import numpy as np
from scipy.optimize import minimize

from utils.features import prepare_for_training
from utils.hypothesis import sigmoid


class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        data_processed, features_mean, features_deviation = prepare_for_training(
            data, polynomial_degree, sinusoid_degree, normalize_data
        )
        self.data = data_processed
        self.labels = labels

        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        self.num_examples, self.num_features = self.data.shape

        self.unique_labels = np.unique(self.labels)  # labels里有多少个不同值(分为几类)
        self.num_unique_labels, *_ = self.unique_labels.shape
        self.theta = np.zeros((self.num_unique_labels, self.num_features))  # 初始化theta

    def train(self, num_iterations=1000):
        cost_histories = []  # 保存损失值
        for label_index, label in enumerate(self.unique_labels):  # 多分类可以看做是多个二分类
            current_theta = np.copy(self.theta[label_index].reshape(self.num_features, 1))  # 找到属于当前类别的theta值
            current_labels = (self.labels == label).astype(float)  # 二分类，属于当前标签则为1， 不属于则为0
            current_theta, cost_history = self.gradient_descent(
                self.data, current_labels, current_theta, num_iterations
            )
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)
        return self.theta, cost_histories

    def gradient_descent(self, data, labels, theta, num_iterations):
        """
        梯度下降
        :param data: 数据
        :param labels: 标签类别
        :param theta: theta值
        :param num_iterations: 迭代次数
        :return:
        """
        cost_history = []
        num_features = data.shape[1]
        # 优化
        result = minimize(
            lambda current_theta: self.cost_function(data, labels, current_theta.reshape((num_features, 1))),  # 优化目标函数
            theta,  # 待优化的权重系数
            method='CG',  # 优化策略
            jac=lambda current_theta: self.gradient_step(data, labels, current_theta.reshape((num_features, 1))),  # 优化公式
            callback=lambda current_theta: cost_history.append(self.cost_function(data, labels, current_theta.reshape((num_features, 1)))),
            options={'maxiter': num_iterations}
        )
        if not result.success:
            print('执行失败')
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history

    def cost_function(self, data, labels, theta):
        """
        计算损失值
        :param data:
        :param labels:
        :param theta:
        :return:
        """
        num_examples = data.shape[0]
        labels_predict = self.hypothesis(data, theta)
        # 属于当前类别的损失值
        is_current_label = np.dot(labels[labels == 1].T, np.log(labels_predict[labels == 1]))

        # 不属于当前类别的损失值
        not_current_label = np.dot(1 - labels[labels == 0].T, np.log(1 - labels_predict[labels == 0]))

        cost = (-1/num_examples) * (is_current_label + not_current_label)
        return cost

    @staticmethod
    def hypothesis(data, theta):
        """
        计算预测值
        :param data:
        :param theta:
        :return:
        """
        predict = np.dot(data, theta)  # 计算出预测值
        # 逻辑回归需将预测值映射到sigmoid函数
        predict = sigmoid(predict)
        return predict

    def gradient_step(self, data, labels, theta):
        """
        梯度下降每一步的结果计算
        :param data:
        :param labels:
        :param theta:
        :return:
        """
        num_examples = data.shape[0]
        predict = self.hypothesis(data, theta)
        label_diff = predict - labels
        gradient = (1/num_examples) * (np.dot(data.T, label_diff))
        return gradient.T.flatten()  # 转换成行向量

    def predict(self, data):
        """
        预测
        :param data:
        :return:
        """
        num_examples = data.shape[0]
        data_processed, *_ = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data
        )
        predictions = self.hypothesis(data_processed, self.theta.T)
        max_prob_index = np.argmax(predictions, axis=1)  # 找出概率值最大的所在位置
        predict_cls = np.empty(max_prob_index.shape, dtype=object)  # 初始化预测标签
        for index, label in enumerate(self.unique_labels):
            predict_cls[max_prob_index == index] = label
        return predict_cls.reshape((num_examples, 1))
