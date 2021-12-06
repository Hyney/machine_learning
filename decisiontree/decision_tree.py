from math import log
import pickle

from decisiontree.tree_visualize import create_plot


def create_data_set():
    data_set = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no']
    ]
    feature_names = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return data_set, feature_names


def majority_count(class_list):
    """
    获取最多的标签值
    :param class_list:
    :return:
    """
    class_count = {}
    for value in class_list:
        class_count[value] = class_count.get(value, 0) + 1
    return sorted(class_count.items(), key=lambda data: data[1], reverse=True)[0][0]


def calc_shannon_ent(data):
    """
    计算熵值
    :param data:
    :return:
    """
    num_examples = len(data)  # 样本数量
    label_count = {}
    for vec in data:
        current_class_value = vec[-1]
        label_count[current_class_value] = label_count.get(current_class_value, 0) + 1
    shannon_ent = 0
    for key in label_count:
        prob = label_count[key] / num_examples
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def choose_best_feature(data):
    """
    获取决策分类的最佳特征
    :param data:
    :return:
    """
    num_features = len(data[0]) - 1  # 特征个数, 本例数据包含分类值， 所以减1
    base_info_entropy = calc_shannon_ent(data)   # 基础熵值
    best_info_gain = 0   # 信息增益
    best_feature_index = -1
    for i in range(num_features):
        feature_list = [example[i] for example in data]   # 拿到某特征列的值
        unique_values = set(feature_list)  # 对值进行去重
        condition_entropy = 0
        for val in unique_values:
            sub_data_set = split_data_set(data, i, val)  # 索引为i的特征列中，其值为val的数据(不包含索引i的特征列)
            prob = len(sub_data_set) / float(len(data))  # 该数据占总样本数据集的比例
            condition_entropy += prob * calc_shannon_ent(sub_data_set)  # 条件熵
        info_gain = base_info_entropy - condition_entropy  # 计算信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain   # 更新信息增益
            best_feature_index = i
    return best_feature_index


def split_data_set(data, feature_index, feature_val):
    """
    数据集切分
    :param data: 总样本数据集
    :param feature_index: 特征索引
    :param feature_val: 要筛选的特征值
    :return:
    """
    split_data = []  # 结果数据集
    for example in data:
        # 若某样本的feature_index列的值等于要筛选的特征值，就把该样本去除掉这个特征列后的数据放入结果数据集
        if example[feature_index] == feature_val:
            reduced_feature = example[: feature_index]
            reduced_feature.extend(example[feature_index+1:])
            split_data.append(reduced_feature)
    return split_data


def create_tree(data_set, feature_names, feature_labels: list):
    """
    构造决策树
    :param data_set: 数据集
    :param feature_names: 特征名
    :param feature_labels: 决策分类特征顺序
    :return:
    """
    class_list = [example[-1] for example in data_set]   # 拿出传入的数据集的所有标签值
    if class_list.count(class_list[0]) == len(class_list):  # 如果标签集都为同一标签，则不需要再进行划分树
        return class_list[0]
    if len(data_set[0]) == 1:   # 数据集特征筛选完毕，只剩下标签
        return majority_count(class_list)
    best_feature_index = choose_best_feature(data_set)  # 选择对分类效果最佳的一个特征的索引
    best_feature_name = feature_names[best_feature_index]
    feature_labels.append(best_feature_name)   # 决策分类选择的特征顺序
    my_decision_tree = {best_feature_name: {}}
    del feature_names[best_feature_index]   # 选择了一个决策分类的特征后, 在剩余的特征中去除该特征数据
    feature_values = [example[best_feature_index] for example in data_set]  # 取出该特征的值
    unique_value = set(feature_values)   # 去重所选特征值
    for value in unique_value:  # 对特征值再做决策分支
        sub_data_set = split_data_set(data_set, best_feature_index, value)  # 数据集拆分，去除已选特征列
        my_decision_tree[best_feature_name][value] = create_tree(sub_data_set, feature_names, feature_labels)
    return my_decision_tree


if __name__ == '__main__':
    data_sets, columns = create_data_set()
    decision_feature_labels = []
    tree = create_tree(data_sets, columns, decision_feature_labels)
    create_plot(tree)
