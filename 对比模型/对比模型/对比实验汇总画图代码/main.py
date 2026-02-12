import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取xlsx文件
df = pd.read_excel('对比模型.xlsx')
#df = pd.read_excel('duibi.xlsx')
# 提取四种对比模型的列数据
#ture_f=df['TR']
TR_data=df['TR']
LSTM_INF_data = df['LSTM-INF']
Trans_data=df['transformer']
Lstm_data=df['LSTM']
CNN_LSTM_data = df['CNN-LSTM']
SVM_data = df['SVM']
CNN_data = df['CNN']

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(TR_data, label='真实值', linestyle='-', linewidth=2)  #
plt.plot(CNN_data, label='CNN', linestyle='-', linewidth=2)  #
plt.plot(SVM_data, label='SVM', linestyle='-.', linewidth=2)  #
plt.plot(Lstm_data, label='LSTM', linestyle=':', linewidth=2)  #
plt.plot(CNN_LSTM_data, label='CNN-LSTM', linestyle='--', linewidth=2)  #
plt.plot(Trans_data, label='transformer', linestyle='-.', linewidth=2)
plt.plot(LSTM_INF_data, label='LSTM-informer', linestyle='-', linewidth=2)  # 使用实线线表示本文模模型
plt.xlabel('时间间隔/15min',fontsize=20)#这些名字都能改
plt.ylabel('功率/MW',fontsize=20)
plt.xticks(range(0, len(df), 200), fontsize=15)#x轴分辨率
plt.legend(fontsize=15)  # 调整图例的字体大小
plt.show()
