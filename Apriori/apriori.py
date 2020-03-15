

def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data):
    """
    构造一项集
    :param data:
    :return:
    """
    result = []
    for transaction in data:
        for item in transaction:
            if not [item] in result:
                result.append([item])
    result.sort()
    return list(map(frozenset, result))   # frozenset --> 生成一个不可变集合


def scan_data(data_set, ck, min_sup):
    """
    扫描项集, 计算支持度
    :param data_set: 数据集
    :param ck: 项集
    :param min_sup: 最小支持度
    :return:
    """
    result = {}
    for tid in data_set:
        for can in ck:
            if can.issubset(tid):
                result[can] = result.get(can, 0) + 1

    num_items = float(len(data_set))
    ret_list = []     # 存储支持度大于最小支持度的项集
    support_data = {}  # 存储每个项集的支持度
    for key in result:
        support = result[key] / num_items   # 求支持度
        if support >= min_sup:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(item_set: list, k: int):
    """
    生成项集组合
    :param item_set: 项集
    :param k:
    :return:
    """
    result = []
    length = len(item_set)
    for i in range(length):
        for j in range(i+1, length):
            first = list(item_set[i])[:-1]  # 构造下一项集，只需要比较当前项集除最后一项外的其他项是否相同,相同则合并(因为已经排除了支持度较小的项集->即关联性较弱)，如 01, 02, 12 -> 012
            second = list(item_set[j])[:-1]
            if first == second:
                result.append(item_set[i] | item_set[j])
    return result


def apriori_get(data_set, min_support=0.5):
    """
    挖掘频繁项集
    :param data_set:
    :param min_support:
    :return:
    """
    c1 = create_c1(data_set)  # 构建一项集
    item_set_1, support_data = scan_data(data_set, c1, min_support)  # 扫描项集, 返回过滤掉支持度小于最小支持度的项集(一项集)
    item_set_all = [item_set_1]  # 用于存储所有项集, 所有一项集，二项集....
    k = 2
    while len(item_set_all[-1]) > 0:
        item_set = apriori_gen(item_set_all[-1], k)
        ret_list, support = scan_data(data_set, item_set, min_support)  # 构造下一项集, 例：一项集构造二项集，二项集构造三项集
        support_data.update(support)  # 更新上一步得出的项集的支持度
        item_set_all.append(ret_list)   # 将得出的项集添加到
        k += 1
    return item_set_all, support_data


def rules_from_freq(frequency_set, support_data, H, rules, min_conf):
    """

    :param frequency_set: 频繁项集
    :param support_data: 支持度
    :param H: 非频繁项集的元素构成的列表
    :param rules: 关联规则
    :param min_conf: 最小置信度
    :return:
    """
    length = len(H[0])
    while len(frequency_set) > length:
        H = cal_conf(frequency_set, H, support_data, rules, min_conf)
        if len(H) > length:
            apriori_gen(H, length+1)
            length += 1
        else:
            break


def cal_conf(frequency_set, H, support_data, rules, min_conf):
    """
    计算置信度
    :param frequency_set: 频繁项集
    :param H: 频繁项集的元素构成的列表
    :param support_data: 所有项集的支持度
    :param rules: 规则
    :param min_conf: 最小置信度
    :return:
    """
    pruned_h = []
    for con_seq in H:
        conf = support_data[frequency_set] / support_data[frequency_set - con_seq]
        if conf >= min_conf:
            print(frequency_set - con_seq, '---->', con_seq, 'conf:', conf)
            rules.append((frequency_set - con_seq, con_seq, conf))
            pruned_h.append(con_seq)
    return pruned_h


def gen_rules(item_set_all, support_data, min_conf=0.6):
    """
    生成关联规则
    :param item_set_all: 所有频繁项集
    :param support_data: 支持度
    :param min_conf: 最小置信度
    :return:
    """
    rules = []
    for i in range(1, len(item_set_all)):   # 两两之间的关系，所以从1开始
        for frequency_set in item_set_all[i]:
            H = [frozenset([item]) for item in frequency_set]
            rules_from_freq(frequency_set, support_data, H, rules, min_conf)


if __name__ == '__main__':
    data = load_data_set()
    L, support = apriori_get(data)
    i = 0
    for freq in L:
        print('项数:', i+1, ':', freq)
        i += 1
    gen_rules(L, support)
