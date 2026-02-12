import pandas as pd
from sklearn.ensemble import RandomForestRegressor # 随机森林回归预测
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump, load


# 加载数据集
# 训练集
train_set = load('train_set')
train_label = load('train_label')
# 测试集
test_set = load('test_set')
test_label = load('test_label')
# 构建随机森林模型并进行训练：
# 构建随机森林 预测模型

# 设置参数
params = {
    'n_estimators': 200,  # 设置树的棵树
    'max_depth': 4,  # 设置树的深度
    'min_samples_leaf': 3 # 设置树的最小叶子树
}


model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_leaf=params['min_samples_leaf'])

# 模型训练
model.fit(train_set, train_label)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
# 进行预测并评估模型性能：
# 进行预测
y_pred = model.predict(test_set)


# 反归一化处理
# 使用相同的均值和标准差对预测结果进行反归一化处理
# 反标准化
scaler  = load('scaler ')
test_label = scaler.inverse_transform(test_label)
y_pred = y_pred.reshape(-1, 1)
y_pred = scaler.inverse_transform(y_pred)

score = r2_score(test_label, y_pred)
print('*'*50)
print(' 随机森林 模型分数--R^2:', score)


print('*'*50)
# 测试集上的预测误差
test_mse = mean_squared_error(test_label, y_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_label, y_pred)
print('测试数据集上的均方误差--MSE: ',test_mse)
print('测试数据集上的均方根误差--RMSE: ',test_rmse)
print('测试数据集上的平均绝对误差--MAE: ',test_mae)
# 可视化结果
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')


plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test_label, label='真实值',color='orange')  # 真实值
plt.plot(y_pred, label='随机森林预测值',color='green')  # 预测值
plt.legend()
plt.show()