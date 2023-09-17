# -*- coding: utf-8 -*-
# @Author  : YWENROU
# @Time    : 2023/9/17 9:33
# @Describe:多元线性回归描述

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# 1.导入数据
dataset = pd.read_csv('../data/50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# 2. 将第 3 列文本数据转换为数字
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])

# 创建 OneHotEncoder 对象并对第 3 列进行独热编码
# 在避免虚拟变量陷阱时，我们应该使用 OneHotEncoder 的 drop='first' 参数来指定在独热编码时丢弃第一个虚拟变量。
onehot_encoder = OneHotEncoder(categories='auto', drop='first')
X_encoded = onehot_encoder.fit_transform(X[:, 3].reshape(-1, 1)).toarray()
# 将独热编码后的特征拼接到原特征矩阵中
X = np.concatenate((X[:, :3], X_encoded, X[:, 4:]), axis=1)

# 3.将标签转变为Label 编码
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# 4.划分训练集及测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 5. 训练集使用简单线性回归模型来训练
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

# 6.可视化
# 6.1 训练集结果可视化
plt.scatter(X_train[:, 0], Y_train, edgecolors='red')  # 选择第一个特征 X1
plt.plot(X_train[:, 0], regressor.predict(X_train), color='blue')  # 绘制回归线
plt.title("Train DataSet")
plt.show()
# 6.2 测试集可视化
plt.scatter(X_test[:, 0], Y_test, edgecolors='red')  # 选择第一个特征 X1
plt.plot(X_test[:, 0], regressor.predict(X_test), color='blue')  # 绘制回归线
plt.title("Test DataSet")
plt.show()







