import numpy as np


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        # 1、先随机选择K个中心点
        centroids = self.centroids_init(self.data, self.num_clusters)   # 质心
        # 2、开始训练
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # 3、得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centroids_ids = self.centroids_find_closest(self.data, centroids)
            # 4、将得出的新中心点进行更新
            centroids = self.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)
        return closest_centroids_ids, centroids

    @staticmethod
    def centroids_init(data, num_clusters):
        """
        随机设置K个特征空间内的点作为初始的聚类中心点
        :param data: 数据源
        :param num_clusters: 聚簇数
        :return:
        """
        num_example = data.shape[0]
        random_ids = np.random.permutation(num_example)
        centroids = data[random_ids[:num_clusters], :]
        return centroids

    def centroids_find_closest(self, data, centroids):
        """
        计算每个样本点到中心点的距离，求距离最近的一个
        :param data:
        :param centroids:
        :return:
        """
        num_examples = self.data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):  # 每个样本计算一次
            distance = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):  # 计算样本到每个中心点的距离
                distance_diff = data[example_index, :] - centroids[centroid_index, :]
                distance[centroid_index] = np.sum(distance_diff ** 2)
            closest_centroids_ids[example_index] = np.argmin(distance)  # 找出最小距离的位置
        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clusters):
        """
        接着对已聚类的样本，重新计算出每个聚类的新中心点(平均值)
        :param data:
        :param closest_centroids_ids:
        :param num_clusters:
        :return:
        """
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters, num_features))
        for centroid_id in range(num_clusters):
            closest_ids = closest_centroids_ids == centroid_id  # 如果计算得出的新中心点与原中心点一样。则结束，否则重新进行第二步过程
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)  # 接着对已聚类的样本，重新计算出每个聚类的新中心点(平均值)
        return centroids
