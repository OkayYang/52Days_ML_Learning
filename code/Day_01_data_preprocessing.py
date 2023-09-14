# -*- coding: utf-8 -*-
# @Author  : YWENROU
# @Time    : 2023/9/14 15:53
# @Function:

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. 导入数据
dataset = pd.read_csv('../data/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# 2. 处理缺失数据
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# 3. 解析分类数据
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

# 将分类特征进行独热编码
one_hot_encoder = OneHotEncoder(sparse=False)
X_categorical = one_hot_encoder.fit_transform(X[:, 0].reshape(-1, 1))

# 将独热编码后的特征与其他特征合并
X = np.concatenate((X_categorical, X[:, 1:]), axis=1)

label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# 4. 拆分数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 5. 特征量化
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)