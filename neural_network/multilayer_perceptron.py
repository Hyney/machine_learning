import numpy as np
from machine_learning.utils.features import prepare_for_training
from machine_learning.utils.hypothesis import sigmoid, sigmoid_gradient
# from sklearn.neural_network import multilayer_perceptron


class MultilayerPerceptron:  # 多层感知机
    """手写数字识别, (28, 28, 1)"""
    def __init__(self, data, labels, layers, normalize_data=False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed
        self.labels = labels
        # 神经网络的层次结构一般为：输入层, 中间层(隐藏层)(可以有多个), 输出层。
        # 在一个神经网络中，输入层和输出层的节点数往往是固定的。
        # 对于输入层，每个样本特征即一个节点, 即输入层节点数 = 样本特征数(向量维数)
        # 对于输出层，每个类别代表一个节点 即输出层节点数 = 分类类别个数。层与层之间的映射关系系数即权重系数。
        # 对于隐藏层，有三种：(输入节点数 + 输出节点数)开方 + α∈(1,10); 2: (输入节点数 * 输出节点数)开方; 3: log输入节点数
        self.layers = layers   # 神经网络的每层的节点个数， 本实验中，样本特征个数28*28*1=784个，分类类别为0-9共10个类别，因此，输入层784个结点， 输出层10个结点，设中间层25个结点。
        self.normalize_data = normalize_data
        self.thetas = self.thetas_init(layers)  # 存放神经网络训练过程中需要的权重系数, 本实验有两个映射关系: 输入层 --> 中间层   , 中间层 --> 输出层。即需要两个映射关系系数(权重系数)

    def predict(self, data):
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]

        predictions = self.feedforward_propagation(data_processed, self.thetas, self.layers)

        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    def train(self, max_iterations=1000, alpha=0.1):
        """

        :param max_iterations:
        :param alpha: 学习率
        :return:
        """
        unroll_thetas = self.thetas_unroll(self.thetas)
        (optimized_theta, cost_history) = self.gradient_descent(self.data, self.labels, unroll_thetas,
                                                                                self.layers, max_iterations, alpha)

        self.thetas = self.thetas_roll(optimized_theta, self.layers)
        return self.thetas, cost_history

    @classmethod
    def gradient_descent(cls, data, labels, unroll_theta, layers, max_iterations, alpha):
        """
        梯度更新
        :param data: 样本数据
        :param labels: 样本分类标签
        :param unroll_theta: 权重系数--行矩阵
        :param layers: 神经网络层级结构
        :param max_iterations:迭代次数
        :param alpha: 学习率
        :return:
        """
        optimized_theta = unroll_theta
        cost_history = []
        thetas = cls.thetas_roll(unroll_theta, layers)
        for _ in range(max_iterations):
            cost = cls.cost_function(data, labels, thetas, layers)
            cost_history.append(cost)
            theta_gradient = cls.gradient_step(data, labels, thetas, layers)
            optimized_theta -= alpha * theta_gradient
        return optimized_theta, cost_history

    @classmethod
    def gradient_step(cls, data, labels, thetas, layers):
        """
        单步梯度下降
        :param data:
        :param labels:
        :param thetas: 行矩阵权重系数还原后的权重系数
        :param layers:
        :return:
        """
        theta_gradient = cls.back_propagation(data, labels, layers, thetas)
        theta_unroll_gradient = cls.thetas_unroll(theta_gradient)
        return theta_unroll_gradient

    @staticmethod
    def back_propagation(data, labels, layers, thetas):
        """
        反向传播，更新梯度
        :param data:
        :param labels:
        :param layers:
        :param thetas:
        :return:
        """
        num_layers = len(layers)  # 神经网络的层级结构
        num_examples, num_features = data.shape  # 样本数据及特征
        num_labels = layers[-1]  # 分类类别数

        deltas = {}    # 初始化梯度变化值
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            deltas[layer_index] = np.zeros((out_count, in_count+1))

        for example_index in range(num_examples):
            layer_inputs = {}  # 用于存放每层的输入
            layer_activations = {}  # 用于存放每层的输出
            layer_activation = data[example_index, :].reshape(1, num_features)   # 获取一个样本数据。 输入层输入。本实验(1, 785)
            layer_activations[0] = layer_activation
            layer_inputs[0] = layer_activation
            for layer_index in range(num_layers - 1):  # 逐层计算
                layer_theta = thetas[layer_index]  # 得到当前层的权重系数
                layer_input = np.dot(layer_activation, layer_theta.T)  # 当前层的输出，下一层的输入(1, 25) (1, 10)
                layer_activation = np.hstack((np.array([[1]]), sigmoid(layer_input)))  # 考虑偏置项
                layer_inputs[layer_index + 1] = layer_input
                layer_activations[layer_index + 1] = layer_activation
            output_layer_activation = layer_activation[:, 1:]   # 输出层输出结果，去掉偏置项

            delta = {}  # 用于存放差异值
            bitwise_label = np.zeros((1, num_labels))  # (1, 10)
            bitwise_label[0][labels[example_index][0]] = 1
            delta[num_layers - 1] = output_layer_activation - bitwise_label   # 预测值和真实值之间的差异

            for layer_index in range(num_layers - 2, 0, -1):  # 从输出层的前一层开始到输入层上一层为止，逐层更新；输出层位置: num_layers -1
                layer_theta = thetas[layer_index]  # (10, 26) 获取权重系数：1、隐层n -> 输出层; 2、隐层n-1 -> 隐层n, .....
                next_delta = delta[layer_index + 1]
                layer_input = layer_inputs[layer_index]
                layer_input = np.hstack((np.array([[1]]), layer_input))
                delta[layer_index] = np.dot(next_delta, layer_theta) * sigmoid_gradient(layer_input)
                delta[layer_index] = delta[layer_index][:, 1:]  # 过滤掉偏置项

            for layer_index in range(num_layers - 1):
                layer_delta = np.dot(delta[layer_index + 1].T, layer_activations[layer_index])
                deltas[layer_index] += layer_delta

        for layer_index in range(num_layers - 1):
            deltas[layer_index] = deltas[layer_index] * (1 / num_examples)

        return deltas

    @classmethod
    def cost_function(cls, data, labels, thetas, layers):
        """
        损失函数
        :param data:
        :param labels: 分类标签。本实验用手写数字，则标签值为0-9的某个数字。
        :param thetas:
        :param layers:
        :return:
        """
        num_layers = len(layers)
        num_examples = data.shape[0]   # 样本量
        num_labels = layers[-1]    # 分类数
        # 前向传播走一次
        predictions = cls.feedforward_propagation(data, thetas, layers)
        bitwise_labels = np.zeros((num_examples, num_labels))   # 分类结果(输出标签)。初始化为0矩阵，每列值表示是否属于0-9类别, 1: True, 0: False
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1   # 1、找到样本数据在输出标签中行位置, 2、再找输出标签中列位置(分类标签的值 对应的位置) 的值 置为1
        y_is_set = np.sum(np.log(predictions[bitwise_labels == 1]))
        y_is_not_set = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
        cost = -1 / num_examples * (y_is_set + y_is_not_set)
        return cost

    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        """
        前向传播，得出输出层结果。
        :param data:
        :param thetas:
        :param layers:
        :return:
        """
        num_examples = data.shape[0]   # 获取样本量。本实验数据为(1700, 785), 预处理时已经增加了一列偏置项
        in_layer_activation = data

        for layer_index in range(len(layers) - 1):  # 逐层计算
            theta = thetas[layer_index]
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))
            # 偏置项考虑: 增加一列偏置项, 下一层的权重系数考虑了偏置项，不增加会出错
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation  # 输入层的输出即隐层的输入。
        return in_layer_activation[:, 1:]  # 因为in_layer_activation第一列为偏置项，前向传播结果(输出层结果)不需要

    @staticmethod
    def thetas_unroll(thetas):
        """
        将thetas（一个(25, 785)的矩阵转换成一个行矩阵，用于theta更新, 更新后再将其还原
        :param thetas: 权重参数
        :return:
        """
        unroll_thetas = np.array([])
        for theta_layer_index in range(len(thetas)):  # 将权重参数矩阵转换成行矩阵
            unroll_thetas = np.hstack((unroll_thetas, thetas[theta_layer_index].flatten()))  # 将所有层间的权重系数拼接在一起
        return unroll_thetas

    @staticmethod
    def thetas_roll(unroll_thetas, layers):
        """
        将转换成的行矩阵还原会转换前的矩阵形式
        :param unroll_thetas: 转换后的权重参数(行矩阵)
        :param layers: 神经网络层级结构
        :return:
        """
        start = 0
        thetas = {}
        for layer_index in range(len(layers) - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            thetas_col = in_count + 1   # 转换成行矩阵前，矩阵的列数。+1 是初始化时考虑了偏置项
            thetas_row = out_count      # 转换成行矩阵前，矩阵的行数
            thetas_volume = thetas_row * thetas_col  # 转换成的行矩阵的大小
            thetas[layer_index] = unroll_thetas[start:start+thetas_volume].reshape(thetas_row, thetas_col) # 将行矩阵取出后，重塑
            start += thetas_volume
        return thetas

    @staticmethod
    def thetas_init(layers):
        """
        初始化层次间的权重参数: 输入层：784个节点, 中间层：25个节点, 输出层：10个节点
        层级映射关系：y = XW  其中W为权重系数(映射系数)(通常将一组权重系数以一维向量表示(),因此在矩阵乘法运算时该W = W.T).注意此处输入输出，系数均为矩阵。如(以一个样本数据为例): 输入层 -> 中间层, 转换成数学公式即：(1, 25) = (1, 784) * W.T  => W为(25, 784)的矩阵 => 如此
        :param layers: 神经网络层级结构
        :return:
        """
        num_layers = len(layers)   # 神经网络的层次数
        thetas = {}
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]    # 获取相对输入层的节点个数---此处相对，即两两映射层之间--方便自己理解而引入，如：输入层 --> 中间层，这二者间相对输入层即输入层，相对输出层为中间层
            out_count = layers[layer_index+1]  # 获取相对输出层的节点个数
            thetas[layer_index] = np.random.randn(out_count, in_count+1) * 0.05   # 乘以0.05是让初始权重系数略小一些， 加1是因为这里需要考虑偏置项。偏置项的个数与相对输出层的节点个数相同。权重参数w：(25, 785), 25行785列, 最后一列为偏置项
        return thetas
