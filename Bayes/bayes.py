# 垃圾邮件过滤实例
from pathlib import Path
import random
import re

import numpy as np


def text_parse(input_string):
    """
    单词切分
    :param input_string:
    :return:
    """
    list_tokens = re.split(r'\W+', input_string)
    return [token.lower() for token in list_tokens if len(list_tokens) > 2]


def open_file(file_path, label, doc_list=None, class_list=None):
    """
    读取文件并处理
    :param file_path: 文件夹路径
    :param label: 邮件标签, 1 表示垃圾邮件
    :param doc_list: 邮件单词列表
    :param class_list: 邮件标签列表
    :return:
    """
    if not doc_list:
        doc_list = []
        class_list = []
    path = Path(file_path)
    for file in path.iterdir():
        word_list = text_parse(file.open('r').read())
        doc_list.append(word_list)
        class_list.append(label)  # 1 表示垃圾邮件
    return doc_list, class_list


def create_vocab_list(doc_list):
    """
    构建语料表
    :param doc_list:
    :return:
    """
    vocab_set = set([])
    for doc in doc_list:
        vocab_set |= set(doc)  # 集合并集, 去重
    return list(vocab_set)


def set_word2vec(vocab_list, input_set):
    """
    语料表转换为向量
    :param vocab_list:
    :param input_set: 邮件单词列表
    :return:
    """
    vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:   # 单词在语料表中则标记为1
            vec[vocab_list.index(word)] = 1
    return vec


def train_nb(train_mat: np.array, train_class:np.array):
    """

    :param train_mat:
    :param train_class:
    :return:
    """
    num_train_doc = len(train_mat)   # 统计邮件的数量
    num_words = len(train_mat[0])   # 统计邮件的词数
    p_spam = sum(train_class)/float(num_train_doc)   # 训练集中垃圾邮件的概率
    p_0_num = np.ones((num_words))  # 平滑处理, 如: 某词在语料表出现，但并未在垃圾邮件中出现，则概率为0,累乘后总概率为0, 因此需要处理.若未出现，则其概率设为1
    p_1_num = np.ones((num_words))  # 平滑处理,

    p_0_denom = 2
    p_1_denom = 2

    for i in range(num_train_doc):
        if train_class[i] == 1:
            p_1_num += train_mat[i]
            p_1_denom += sum(train_mat[i])
        else:
            p_0_num += train_mat[i]
            p_0_denom += sum(train_mat[i])

    p_1_vec = np.log(p_1_num/p_1_denom)
    p_0_vec = np.log(p_0_num/p_0_denom)
    return p_0_vec, p_1_vec, p_spam


def classify_nb(word_vec, p_0_vec, p_1_vec, p1_class):
    p1 = np.log(p1_class) + sum(word_vec*p_1_vec)
    p0 = np.log(1.0 - p1_class) + sum(word_vec*p_0_vec)
    return 0 if p0 > p1 else 1


def spam():
    """邮件读取并标记"""
    spam_path = './email/spam'
    doc_list, class_list = open_file(spam_path, 1)
    ham_path = './email/ham'
    doc_list, class_list = open_file(ham_path, 0, doc_list, class_list)
    vocab_list = create_vocab_list(doc_list)  # 构建语料表
    train_set = list(range(50))
    test_set = []
    for i in range(10):   # 训练集、测试集划分
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        train_set.pop(rand_index)
        # del train_set[rand_index]

    train_mat = []
    train_class = []
    for doc_index in train_set:
        train_mat.append(set_word2vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p_0_vec, p_1_vec, p1 = train_nb(np.array(train_mat), np.array(train_class))
    error_count = 0
    for docIndex in test_set:
        word_vec = set_word2vec(vocab_list, doc_list[docIndex])
        if classify_nb(np.array(word_vec), p_0_vec, p_1_vec, p1) != class_list[docIndex]:
            error_count += 1
    print('当前10个测试样本，错了：', error_count)


if __name__ == '__main__':
    spam()

