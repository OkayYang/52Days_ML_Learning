# -*- coding: utf-8 -*-
# @Author  : YWENROU
# @Time    : 2023/9/16 10:42
# @Function:
# @Describe:简单线性回归描述


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据预处理
dataset = pd.read_csv('../data/studentscores.csv')
X = dataset.iloc[:,:1].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# 2.训练集使用简单线性回归模型来训练
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 3.预测结果
Y_pred = regressor.predict(X_test)

# 4.可视化
# 4.1 训练集结果可视化
plt.scatter(X_train,Y_train,edgecolors='red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.title("Train DataSet")
plt.show()
# 4.2 测试集可视化
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.title("Test DataSet")
plt.show()



