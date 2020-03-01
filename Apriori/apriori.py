

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
    return list(map(frozenset, result))

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
    for key in result:
        support = result[key] / num_items   # 求支持度
        if support >= min_sup:
            pass


if __name__ == '__main__':
    data = load_data_set()
    print(create_c1(data))