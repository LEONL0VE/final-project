import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

# 读取数据
original_data = pd.read_csv('feng2019.csv')
print(original_data.shape)
original_data.head()
# 取风速数据
winddata = original_data['Target'].tolist()
winddata = np.array(winddata) # 转换为numpy
# 可视化
plt.figure(figsize=(15,5), dpi=100)
plt.grid(True)
plt.plot(winddata, color='green')
plt.show()
# 制作数据集和标签
import torch
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler


original_data = original_data.iloc[0:, :]

# 1. 输入训练集  变量： FR_torque , FR_angular, Chassis_velocity (x),  Chassis_velocity (z),  Slope
var_data =  original_data[['A', 'B', 'C',
                           'D', 'E', 'F']]
# 转为 numpy 二维矩阵
var_data = var_data.values

# 2. 对应y值标签为：
ylable_data =  original_data[['Target']]
# 转为 numpy 二维矩阵
ylable_data = ylable_data.values


# 归一化处理
# 使用标准化（z-score标准化）
scaler = StandardScaler()
var_data = scaler.fit_transform(var_data)
ylable_data = scaler.fit_transform(ylable_data)
# 保存 归一化 模型
dump(scaler, 'scaler')



# 数据集制作
def make_wind_dataset(var_data, y_data, split_rate = [0.8, 0.2]):
    '''
        参数:
        var_data   : 输入训练集  变量
        y_data     : 输入y值标签  变量
        split_rate : 训练集、测试集划分比例

        返回:
        train_set: 训练集数据
        train_label: 训练集标签
        test_set: 测试集数据
        test_label: 测试集标签
    '''
    # 第一步，划分数据集
    #序列数组
    sample_len = var_data.shape[0] # 样本总长度
    train_len = int(sample_len*split_rate[0])  # 向下取整
    # 变量数据 划分训练集、测试集
    train_set = var_data[:train_len, :] # 训练集
    test_set = var_data[train_len:, :]  # 测试集
    # y标签 划分训练集、测试集
    train_label = y_data[:train_len, :] # 训练集
    test_label = y_data[train_len:, :]  # 测试集

    return train_set, train_label, test_set, test_label

# 训练集、测试集划分比例
split_rate = [0.8, 0.2]

# 制作数据集
train_set, train_label, test_set, test_label = make_wind_dataset(var_data, ylable_data, split_rate)
# 保存数据
dump(train_set, 'train_set')
dump(train_label, 'train_label')
dump(test_set, 'test_set')
dump(test_label, 'test_label')
print('数据 形状：')
print(train_set.shape, train_label.shape)
print(test_set.shape, test_label.shape)