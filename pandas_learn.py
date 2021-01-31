import pandas as pd

# 创建DataFrame结构的数据
data = {'country': ['aaa', 'bbb', 'ccc'], 'population': [10, 20, 30]}
df = pd.DataFrame(data)

# 读取csv文件数据

df = pd.read_csv('./data/titanic_train.csv')

df.head()  # 读取前几条数据, 默认为前5条

df.info()  # 返回当前数据的信息

df.index  # 索引

df.columns  # 列名

# 获取某列的值
age = df['Age']
type(age)  # pandas.core.series.Series
age.index
age.values


# 拷贝数据
a = df.copy()

# 设置索引
a.set_index('Name', inplace=True)  # 设置name列为索引列, inplace: 是否就地修改, 默认为False
print(a['Age'][:5])


# DataFrame获取数据的基本统计特性
df.describe()

# DataFrame定位数据的方式
    # 1、loc 用label索引定位
    # 2、iloc 用position索引定位

df.iloc[0]
df.iloc[:5]  # 切片
df.iloc[:5, 1:3]  # 前5行, 第2, 3列的数据

df.loc['Braund, Mr. Owen Harris']  # 获取name为Braund, Mr. Owen Harris的数据
df.loc['Braund, Mr. Owen Harris', 'Fare']  # 获取某人的某个属性的数据
df.loc['Braund, Mr. Owen Harris':'Allen, Mr. William Henry']  #切片

df.loc['Braund, Mr. Owen Harris', 'Fare'] =  7.25  # 赋值


# bool类型的索引
df[df['Fare'] > 40]  # 获取船票价格大于40的人的数据
df[df['Sex'] == 'male']  # 获取男性的所有数据
df[(df['Sex'] == 'male')&(df['Fare']>40)]  # 获取男性所有船票价格大于40的人的数据
df[(df['Fare'] > 50) | (df['Fare'] < 30)]  # 获取船票价格大于50或者小于30的人的数据
df.loc[~(df['Sex'] == 'male')]  # 获取所有非男性的人的数据
df.loc[df['Sex'] == 'male', 'Age'].mean()  #统计男性的平均年龄


# 分组
df.groupby('Sex').mean()  # 按性别分组
df.groupby('Sex')['Age'].mean()  # 男女的平均年龄


# DataFrame数值计算
d = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=['a', 'b'], columns=['A', 'B', 'C'])
d.sum()  # 按列求和
d.sum(axis=1)  # 按行求和
d.median()  # 中值

# 二元统计
df.cov()  # 协方差
df.corr()  # 相关系数

df['Age'].value_counts(ascending=True)  # 统计各年龄的人数
df['Pclass'].value_counts(ascending=True)  # 统计各船舱的人数

df['Age'].value_counts(bins=5)  # 年龄段人数统计

# 索引/列名重命名
d.rename(index={'a': 'f'}, inplace=True)


# DataFrame merge操作
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'], 'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'], 'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']})

pd.merge(left, right, on='key')

# 多个字段
left = pd.DataFrame(
    {
        'key1': ['K0', 'K1', 'K2', 'K3'],
        'key2': ['K0', 'K1', 'K2', 'K3'],
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3']
    }
)

right = pd.DataFrame(
    {
        'key1': ['K0', 'K1', 'K2', 'K3'],
        'key2': ['K0', 'K1', 'K2', 'K3'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    }
)
pd.merge(left, right, on=['key1', 'key2'])

# 公共字段值不同的合并
left = pd.DataFrame(
    {
        'key1': ['K0', 'K1', 'K2', 'K3'],
        'key2': ['K0', 'K1', 'K2', 'K3'],
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3']
    }
)

right = pd.DataFrame(
    {
        'key1': ['K0', 'K1', 'K2', 'K3'],
        'key2': ['K0', 'K1', 'K2', 'K4'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    }
)
pd.merge(left, right, on=['key1', 'key2'])  #内连接
pd.merge(left, right, how='outer', on=['key1', 'key2'])  # 外连接
pd.merge(left, right, how='left', on=['key1', 'key2'])  # 左连接


# pandas的显示设置
pd.get_option('display.max_rows')  # 最大显示行数
pd.set_option('display.max_rows', 120)  # 设置最大显示行数为120
pd.get_option('display.max_columns')  # 最大显示列数
pd.get_option('display.max_colwidth')  # 网格最大宽度
pd.get_option('display.precision')  # 精度

# 数据透视表

# 1、各船舱男女性的船票平均价格，平均存活率
df.pivot_table(index=['Sex'], columns=['Pclass'], values=['Fare', 'Survived'])

# 2、各船舱男女性的船票总价, 存活人数
df.pivot_table(index=['Sex'], columns=['Pclass'], values=['Fare', 'Survived'], aggfunc='sum')

# 3、各船舱男女性的存活、死亡数量
df.pivot_table(index=['Pclass', 'Sex'], columns=['Survived'], values='Fare', aggfunc='count')

# 4、各船舱男女数量统计
pd.crosstab(index=df['Pclass'], columns=df['Sex'])
df.pivot_table(index=['Pclass'], columns=['Sex'], values='Name', aggfunc='count')


# 常用操作
data = pd.DataFrame(
    {
        'group': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        'data': [4, 3, 2, 1, 12, 3, 4, 5, 7]
    }
)

data.sort_values(by=['group', 'data'], ascending=[False, True])   # 值排序操作, group升序, data降序


data.assign()