# 05Day-逻辑回归(二)
昨天已经了解了逻辑回归的基本原理，现在用代码实现一下，点击此处下载数据[50_Startups.csv](/assets/ML/Social_Network_Ads.csv)

这个训练数据集是一个名为"Social_Network_Ads.csv"的CSV文件。它包含了社交网络广告的一些用户信息和购买行为。下面是数据集中的列和其含义：

- User ID: 用户ID，用于唯一标识每个用户。
- Gender: 用户的性别，可以是"Male"（男性）或"Female"（女性）。
- Age: 用户的年龄。
- EstimatedSalary: 用户的估计工资。
- Purchased: 用户是否购买了广告（目标变量），表示为0（未购买）或1（购买）。

每一行数据代表一个用户的信息和购买行为。例如，第一行数据表示一个19岁的男性用户，估计工资为19000美元，未购买广告（Purchased为0）。

这个数据集可以用于训练机器学习模型，预测用户是否会购买广告。模型可以根据用户的性别、年龄和估计工资等特征，来预测用户的购买行为，从而帮助广告公司或市场营销团队做出更准确的定向广告投放策略。

## 步骤 1: 数据读取及预处理

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据集
dataset = pd.read_csv('../data/Social_Network_Ads.csv')

# 提取特征和目标变量
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

在这个步骤中，我们导入必要的库并执行以下操作：
- 使用`pandas`库的`read_csv`函数读取数据集。
- 使用`.iloc`方法从数据集中提取特征和目标变量。将特征数据存储在`X`变量中，将目标变量数据存储在`Y`变量中。
- 使用`train_test_split`函数将数据集划分为训练集和测试集，并将划分后的数据分别存储在`X_train`、`X_test`、`y_train`和`y_test`变量中。
- 使用`StandardScaler`类对特征进行缩放，将训练集和测试集的特征数据进行转换。

## 步骤 2: 建立逻辑回归模型

```python
from sklearn.linear_model import LogisticRegression

# 创建逻辑回归模型
classifier = LogisticRegression()
```

在这个步骤中，我们导入`LogisticRegression`类并创建了一个逻辑回归模型。我们将该模型存储在名为`classifier`的变量中。

## 步骤 3: 预测

```python
# 在训练集上训练模型
classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = classifier.predict(X_test)
```

在这个步骤中，我们使用训练集的特征数据和目标变量数据，调用逻辑回归模型的`fit`方法进行训练。然后，我们使用训练好的模型对测试集的特征数据进行预测，并将预测结果存储在`y_pred`变量中。

## 步骤 4: 评估

```python
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
```

在这个步骤中，我们使用`confusion_matrix`函数计算混淆矩阵，用于评估逻辑回归模型的性能。我们将测试集的真实目标变量数据`y_test`和预测结果`y_pred`作为参数传递给`confusion_matrix`函数，并将结果存储在`cm`变量中。混淆矩阵提供了模型的真阳性、真阴性、假阳性和假阴性的数量，可以用于评估模型的准确性和召回率等指标。

## 步骤 5: 可视化

```python
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
```
![](https://cos.ywenrou.cn/blog/images20230919162612.png)
```python
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
```
![](https://cos.ywenrou.cn/blog/images20230919162550.png)