import sklearn
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Sklearn verion is {}".format(sklearn.__version__))

# 导入数据
df = pd.read_csv('data.csv')
#  数据分析
num_size = 0.7;                              # 训练集占数据集比例
outdim = 1;                                  # 最后一列为输出
num_samples = df.shape[0];                  # 样本个数
random_indices = np.random.permutation(num_samples) # 生成随机排列的索引
df = df.iloc[random_indices, :]         # 根据随机排列的索引打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples);  #训练集样本个数
f_ = df.shape[1] - outdim;                   #输入特征维度

#  划分训练集和测试集
P_train = df.iloc[:num_train_s, :f_]
T_train = df.iloc[:num_train_s, f_:]
M = P_train.T.shape[1]

P_test = df.iloc[num_train_s:,:f_]
T_test = df.iloc[num_train_s:, f_:]
N = P_test.T.shape[1]

# 创建模型
regr = make_pipeline(StandardScaler(),LinearSVR(random_state=0, tol=1e-5)) # StandardScaler()自动实现标准化
regr.fit(P_train, T_train)  #训练模型

# 获取相关参数值
print(regr.named_steps['linearsvr'].coef_) #获取线性支持向量回归的系数
print(regr.named_steps['linearsvr'].intercept_) #获取模型中的截距

score_train = regr.score(P_train, T_train)
print("在训练集上的得分：", score_train)

score_test = regr.score(P_test, T_test)
print("在测试集上的得分：", score_test)

# 预测
predict1 = regr.predict(P_train)
print("预测结果：", predict1)
predict2 = regr.predict(P_test)
print("预测结果：", predict2)

# 将训练集与测试集转为数组的形式，以便计算相关指标和绘图使用
T_train_flattened = np.squeeze(T_train)
T_train_array = T_train_flattened.to_numpy()
T_train_array.shape

T_test_flattened = np.squeeze(T_test)
T_test_array = T_test_flattened.to_numpy()
T_test_array.shape

# 计算相关评价指标
# R^2就等于 内置的score()函数
# 用来衡量模型拟合数据的程度，取值范围在0到1之间。R2越接近1，说明模型对数据的拟合度越高
R1 = 1 - np.linalg.norm(T_train_array - predict1) ** 2 / np.linalg.norm(T_train_array - np.mean(T_train_array)) ** 2 #训练集的R^2
R2 = 1 - np.linalg.norm(T_test_array - predict2) ** 2 / np.linalg.norm(T_test_array - np.mean(T_test_array)) ** 2 #测试集的R^2

# MAE
# 预测值与实际值之间差值的平均绝对值。MAE越小，说明模型的预测精度越高
MAE1 = np.sum(np.abs(predict1 - T_train_array)) / M #训练集的MAE
MAE2 = np.sum(np.abs(predict2 - T_test_array)) / N #测试集的MAE

# MBE
# MBE是预测值与实际值之间差值的平均值。MBE为0表示模型的预测结果没有偏差，否则表示存在偏差
MBE1 = np.sum(predict1 - T_train_array) / M #训练集的MBE
MBE2 = np.sum(predict2 - T_test_array) / N #测试集的MBE

# MAPE
# MAPE是预测值与实际值之间百分比差值的平均绝对值
MAPE1 = np.sum(np.abs((predict1 - T_train_array) / T_train_array)) / M #训练集的MAPE
MAPE2 = np.sum(np.abs((predict2 - T_test_array) / T_test_array)) / N #测试集的MAPE

# RMSE
# RMSE是预测值与实际值之间差值的平方的均值的平方根。RMSE越小，说明模型的预测精度越高
RMSE1 = np.sqrt(np.sum((predict1 - T_train_array) ** 2) / M)  # 训练集的RMSE
RMSE2 = np.sqrt(np.sum((predict2 - T_test_array) ** 2) / M)  # 训练集的RMSE

print("训练集数据的R2为：{}".format(R1))
print("测试集数据的R2为：{}".format(R2))
print("训练集数据的MAPE为：{}".format(MAPE1))
print("测试集数据的MAPE为：{}".format(MAPE2))
print("训练集数据的MAE为：{}".format(MAE1))
print("测试集数据的MAE为：{}".format(MAE2))
print("训练集数据的MBE为：{}".format(MBE1))
print("测试集数据的MBE为：{}".format(MBE2))
print("训练集数据的RMSE为：{}".format(RMSE1))
print("测试集数据的RMSE为：{}".format(RMSE2))

#  绘图

# 折线图
plt.plot(range(0, M), T_train_array, 'r-*',linewidth=1, label='train_real')
plt.plot(range(0, M), predict1, 'b-o',linewidth=1,label='train_predict')
plt.legend('train_real', 'predict')
plt.xlabel('Sample projections')
plt.ylabel('Results of projected')
string = "LinearSVR score_train is {}".format(score_train)
plt.title(string)
plt.xlim([-3, M+1])
plt.legend()
# 显示图形
plt.grid() #添加网格线
plt.show()

plt.plot(range(0, N), T_test_array, 'r-*',linewidth=1, label='test_real')
plt.plot(range(0, N), predict2, 'b-o',linewidth=1,label='test_predict')
plt.legend('test_real', 'test_predict')
plt.xlabel('Sample projections')
plt.ylabel('Results of projected')
string = "LinearSVR score_test is {}".format(score_test)
plt.title(string)
plt.xlim([-3, N+1])
plt.legend()
plt.show()

# 散点图
sz = 25
c = 'b'

plt.figure()
plt.scatter(T_train_array, predict1, sz, c)
plt.plot(plt.xlim(), plt.ylim(), '--k')
plt.xlabel('True value of training set')
plt.ylabel('Predict value of training set')
plt.xlim([min(T_train_array), max(T_train_array)])
plt.ylim([min(predict1), max(predict1)])
plt.title('True vs. Predict')
plt.show()

plt.figure()
plt.scatter(T_test_array, predict2, sz, c)
plt.plot(plt.xlim(), plt.ylim(), '--k')
plt.xlabel('True value of test set')
plt.ylabel('Predict value of test set')
plt.xlim([min(T_test_array), max(T_test_array)])
plt.ylim([min(predict2), max(predict2)])
plt.title('True vs. Predict')
plt.show()

# 导入带预测的数据(未来数据)
df2 = pd.read_csv('data_predict.csv')
V_train = df2.iloc[:num_train_s, :f_]
predict3 = regr.predict(V_train)
print("预测结果：", predict3)
data = {'预测结果': predict3}
df3 = pd.DataFrame(data)
df3.to_csv('output.csv', index=False)



