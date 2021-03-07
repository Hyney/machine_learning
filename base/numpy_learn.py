import numpy as np


ary = [1, 2, 3, 4, 5]
ary = np.array(ary)
print(ary)

print(ary[1:3])
print(ary[-2:])

# 矩阵格式(多维形式)
array = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)

print(array.shape)
print(array[0])  # 第一行
print(array[0, 1])  # 第一行第二列的元素
print(array[0, :2])  # 第一行前两列的元素

array[1, 1] = 100  # 将第二行第二列的元素设置为100

array_2 = array  # 注意, (引用, 并未开辟新的内存)

array_2[1, 1] = 5
print(array)

array_3 = array.copy()
array_3[1, 1] = 100
print(array_3)


array = np.arange(0, 100, 10)  # 左闭右开, 等同于range

mask = np.array([0, 1, 0, 1, 0, 1], dtype=bool)   # 指定数据类型
# [False  True False  True False  True]

mask = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)

print(array[mask])  # [10, 30, 40, 70, 80]

print(array > 30)  # [False, False, False, False,  True,  True,  True,  True,  True, True]

idx = np.where(array > 30)  # 获取array中大于30的元素的位置/索引

array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
array_4 = np.asarray(array, dtype=np.int32)  # 修改数据类型
print(array.astype(np.int32))   # 效果同asarray方法


# 排序
array = np.array([[1.5, 1.3, 5.2], [2.1, 3.2, 4.7]])
print(np.sort(array))  # [[1.3 1.5 5.2], [2.1 3.2 4.7]]

print(np.sort(array, axis=0))  # [[1.5 1.3 4.7], [2.1 3.2 5.2]]  # 按列排序

# 数组形状
array = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(array.shape)  # (10,)
array.shape = 2, 5
print(array)  # [[0 1 2 3 4],[5 6 7 8 9]]

array.reshape(shape=(2, 5))  # [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

array = array[:, np.newaxis]
print(array.shape)   # (10, 1)
array = array.squeeze()  # 压缩, 去掉所有空的轴
print(array.shape)  # (10, )


array.shape = 2, 5
array = array.trsnpose()  # 转置
array = array.T  # 转置

# 数组的连接
a = np.array([[23, 32, 42], [12, 13, 40]])
b = np.array([[63, 37, 48], [82, 93, 45]])

c = np.concatenate((a, b))  # 按行(Y轴)拼接 [[23, 32, 42], [12, 13, 40],[63, 37, 48], [82, 93, 45]]
e = np.vstack((a, b))

d = np.concatenate((a, b), axis=1)  # 按列(X轴)拼接
f = np.hstack((a, b))

print(e.flatten())  # 拉平 [23, 32, 42, 63, 37, 48, 12, 13, 40, 82, 93, 45]
print(e.ravel())  # 同flatten方法
#

# 初始化数组
a = np.zeros(3)  # [0., 0., 0.]
b = np.zeros((2, 3))  # [[0., 0., 0.], [0., 0., 0.]]

# 构造初始值为1的数组
c = np.ones(3)  # [1., 1., 1.]

d = c * 8   #  [8., 8., 8.]
#

# 构造空数组
a = np.empty(5)
a.fill(2)  # 用2填充.  [2., 2., 2., 2., 2.]

# 运算
ary = np.array([5, 5])
y = np.array([2, 2])

np.multiply(ary, y)  # 对应位置相乘 [10, 10]
np.dot(ary, y)  # 内积

x = np.array([1, 2, 3])
y = np.array([[2, 3, 4], [3, 2, 1]])
print(x * y)  # [[ 2,  6, 12], [ 3,  4,  3]]


# 逻辑运算
x = np.array([1, 2, 3])
y = np.array([2, 2, 3])
z = np.array([0, 1, 0])
print(x == y)  # [False,  True,  True]
print(np.logical_and(x, y))  # 逻辑与  [ True,  True,  True]
print(np.logical_and(x, z))  # [False,  True, False]
print(np.logical_or(x, z))  # 逻辑或
print(np.logical_not(x))   # 取反


# 随机值

# [0, 1)
print(np.random.rand(2, 3))
# [
#     [0.76961752, 0.16334794, 0.86918388],
#     [0.7836811 , 0.94342924, 0.44252991]
# ]

print(np.random.randint(10, size=(5, 4)))
# [
#     [4, 3, 8, 2],
#     [2, 9, 8, 6],
#     [2, 4, 5, 7],
#     [5, 0, 5, 2],
#     [4, 2, 7, 0]
# ]

# 洗牌(打乱)
a = np.array([1, 2, 3])
np.random.shuffle(a)
print(a)  # [3, 2, 1]
#

# 随机种子
np.random.seed(10)

# 构建一个shape(6, 7, 8)的矩阵, 并找到第100个元素的位置
np.unravel_index(100, (6, 7, 8))

# 对一个5*5的矩阵做归一化
ary = np.random.random((5, 5))
ary_min = ary.min()
ary_max = ary.max()
result = (ary-ary_min)/(ary_max-ary_min)

# 找到两个数组中相同的值
z1 = np.random.randint(0, 10, 10)  # [7, 8, 2, 5, 0, 9, 7, 4, 9, 1]
z2 = np.random.randint(0, 10, 10)  # [3, 2, 2, 8, 7, 5, 2, 0, 9, 4]
np.intersect1d(z1, z2)   # [0, 2, 4, 5, 7, 8, 9]
#

# 得到今天，明天，昨天的日期

