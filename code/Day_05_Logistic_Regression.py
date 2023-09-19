# -*- coding: utf-8 -*-
# @Author  : YWENROU
# @Time    : 2023/9/19 15:25
# @Describe:逻辑回归简单代码

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据集
dataset = pd.read_csv('../data/Social_Network_Ads.csv')

# 提取特征和目标变量
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

# 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

# 创建逻辑回归模型
classifier = LogisticRegression()

# 在训练集上训练模型
classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap

# 绘制训练集结果
X_set, y_set = X_train, y_train

# 生成网格点坐标
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

# 对网格点进行预测
Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

# 绘制等高线图
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))

# 设置坐标轴的范围
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 绘制散点图
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=[ListedColormap(('red', 'green'))(i)], label=j)

# 设置标题和坐标轴标签
plt.title('LOGISTIC (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

# 添加图例
plt.legend()

# 显示图形
plt.show()

# 绘制测试集结果
X_set, y_set = X_test, y_test

# 生成网格点坐标
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

# 对网格点进行预测
Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

# 绘制等高线图
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))

# 设置坐标轴的范围
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 绘制散点图
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=[ListedColormap(('red', 'green'))(i)], label=j)

# 设置标题和坐标轴标签
plt.title('LOGISTIC (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

# 添加图例
plt.legend()

# 显示图形
plt.show()