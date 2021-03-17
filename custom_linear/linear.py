import numpy as np
from utils.features import prepare_for_training


class LinearRegression:

    def __init__(
            self, data, labels, eta=0.01, num_iterations=500,
            polynomial_degree=0, sinusoid_degree=0, normalize_data=True
    ):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        :param data:
        :param labels:
        :param eta: 学习率
        :param num_iterations: 迭代次数
        :param polynomial_degree:
        :param sinusoid_degree:
        :param normalize_data:
        """
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

        self.eta = eta
        self.num_iterations = num_iterations

        self.num_examples, self.num_features = self.data.shape  # 训练数据的数量与特征数
        print(self.data.shape)
        self.theta = np.zeros((self.num_features, 1))
        self._loss = None
        self.all_loss = []

    @property
    def loss(self):
        return self._loss

    def train(self):
        """
        训练模块，执行梯度下降
        """
        self.gradient_descent()
        return self.theta, self.all_loss

    def gradient_descent(self):
        """
        实际迭代模块，会迭代num_iterations次
        """
        for _ in range(self.num_iterations):
            self.gradient_step()
            self._loss_function(self.data, self.labels)
            self.all_loss.append(self._loss)

    def gradient_step(self):
        """
        梯度下降参数更新计算方法，注意是矩阵运算
        :return:
        """
        prediction = self.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - self.eta * (1 / self.num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def _loss_function(self, data, labels):
        """
        损失计算方法
        :param data: 数据
        :param labels: 标签
        :return:
        """
        num_examples = data.shape[0]
        delta = self.hypothesis(self.data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        self._loss = cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        predictions = np.dot(data, theta)
        return predictions

    def calc_loss(self, data, labels):
        """
        获取损失值
        :param data:
        :param labels:
        :return:
        """
        data_processed, *_ = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data
        )

        self._loss_function(data_processed, labels)

    def predict(self, data):
        """
        用训练的参数模型，与预测得到回归值结果
        """
        data_processed, *_ = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data
        )

        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions
