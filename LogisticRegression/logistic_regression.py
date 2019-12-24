import numpy as np
from scipy.optimize import minimize

from machine_learning.utils.features import prepare_for_training
from machine_learning.utils.hypothesis import sigmoid


class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        data_processed, features_mean, features_deviation = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)  # 标签个数(不重复)--多分类标签个数，如二分类，就是0, 1
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        num_unique_labels = self.unique_labels.shape[0]
        self.theta = np.zeros((num_unique_labels, num_features))  # 每个特征对应一个theta, 每个类别对应一组theta值, 以鸢尾花数据为例，theta值是一组3行4列的值

    def train(self, max_iterations=1000):
        cost_histories = []   # 保存损失值
        num_features = self.data.shape[1]
        for index, label in enumerate(self.unique_labels):  # 根据分类类别迭代, 每次迭代过程，对当前类别的theta值进行梯度下降求值
            current_initial_theta = np.copy(self.theta[index].reshape(num_features, 1))  # 找到属于当前类别的初始化theta值
            current_labels = (self.labels == label).astype(float)  # 判断分类标签是否等于给定的标签值; 找到属于当前类别的标签位置，并将标签值置为1，不属于当前类别的置为0
            current_theta, cost_history = self.gradient_descent(self.data, current_labels, current_initial_theta, max_iterations)
            self.theta[index] = current_theta.T   # 优化后的theta值转换为了(num_feature, 1) 的行矩阵， 原始theta值是(num_labels, num_features)的矩阵， 因此在将原theta值更新为优化后的值是，需要将其转换为列矩阵
            cost_histories.append(cost_history)
        return cost_histories

    def gradient_descent(self, data, labels, initial_theta, max_iterations):
        """
        梯度下降迭代过程
        :param data: 数据
        :param labels: 标签
        :param initial_theta: theta值
        :param max_iterations: 迭代次数
        :return:
        """
        cost_history = []
        num_features = data.shape[1]
        result = minimize(
            # 要优化的目标
            lambda current_theta: self.cost_func(data, labels, current_theta.reshape((num_features, 1))),
            # 初始化的的权重系数
            initial_theta,
            # 选择优化策略
            method='CG',
            # 梯度下降迭代计算公式
            jac=lambda current_theta: self.gradient_step(data, labels, current_theta.reshape((num_features, 1))),
            # 记录结果
            callback=lambda current_theta: cost_history.append(self.cost_func(data, labels, current_theta.reshape((num_features, 1)))),
            # 迭代次数
            options={'maxiter': max_iterations}
        )
        if not result.success:
            raise Exception(' can not minimize cost_func.{}'.format(result.message))
        optiminied_theta = result.x.reshape((num_features, 1))
        return optiminied_theta, cost_history

    def cost_func(self, data, labels, theta):
        num_examples = data.shape[0]
        prediction = self.hypothesis(data, theta)
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(prediction[labels == 1]))    # 计算属于某个类别时总的损失值: c = np.dot(a, b); c的第一个元素等于a的第一行的元素乘以b的第一列的对应元素
        # labels是列矩阵，以鸢尾花为例，labels是一个(150, 1)的矩阵, 计算所有样本的总损失值，需对其进行转置(用矩阵的方法来计算，不需要循环求和了)
        y_is_not_set_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - prediction[labels == 0]))
        cost = (-1/num_examples) * (y_is_set_cost + y_is_not_set_cost)
        return cost

    @staticmethod
    def hypothesis(data, theta):
        prediction = sigmoid(np.dot(data, theta))  # 将线性输出映射到sigmoid函数，预测得出属于各类别的概率值
        return prediction

    def gradient_step(self, data, labels, theta):
        num_examples = data.shape[0]
        predictions = self.hypothesis(data, theta)
        label_diff = predictions - labels
        gradients = (1/num_examples) * np.dot(data.T, label_diff)
        return gradients.T.flatten()

    def predict(self, data):
        num_examples = data.shape[0]
        data_processed, features_mean, features_deviation = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, normalize_data=self.normalize_data)
        prob = self.hypothesis(data_processed, self.theta.T)
        prob_index = np.argmax(prob, axis=1)  # 找出行最大值的位置, 即找出预测概率值的最大值的位置
        class_prediction = np.empty(prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[prob_index == index] = label
        return class_prediction.reshape((num_examples, 1))
