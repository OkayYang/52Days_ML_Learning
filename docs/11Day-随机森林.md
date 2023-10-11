
# 11Day-随机森林
![](https://cos.ywenrou.cn/blog/images20231011101240.png)



## 步骤 1: 引入必要的库

首先，让我们引入必要的Python库，这些库将在整个代码中使用。

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## 步骤2：导入数据集

在这一步，我们将导入我们的数据集，该数据集位于文件"Social_Network_Ads.csv"中。数据集通常包含了特征（features）和目标变量（target variable）。你可以在[Social_Network_Ads.csv](/assets/ML/Social_Network_Ads.csv)下载数据集。

这个数据集看起来像这样：

| Age | EstimatedSalary | Purchased |
|-----|-----------------|-----------|
| 19  | 19000           | 0         |
| 35  | 20000           | 0         |
| ... | ...             | ...       |

我们的目标是根据年龄（Age）和估计工资（EstimatedSalary）来预测用户是否购买了某个产品（Purchased）。我们将数据集拆分为特征（X）和目标变量（y）。

```python
# 导入数据集
dataset = pd.read_csv('../data/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
```
## 步骤 3: 数据预处理

### 3.1数据拆分

在这一步骤中，我们将数据集分为两个部分：训练集和测试集。

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

- `train_test_split`：这是Scikit-Learn库的函数，用于将数据集拆分为训练集和测试集。`test_size` 参数设置了测试集的比例。

### 3.2特征缩放

特征缩放是将特征的值缩放到相同的尺度范围，以确保模型的性能不会受到不同尺度特征的影响。

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

- `StandardScaler`：这是Scikit-Learn库的标准化类，用于将特征标准化为均值为0、方差为1的正态分布。

## 步骤 4: 构建随机森林分类器

在这一步骤中，我们构建了一个随机森林分类器。

```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

```
- RandomForestClassifier：这是Scikit-Learn库中的随机森林分类器类，用于构建随机森林模型。
- n_estimators=10：这个参数指定了随机森林中的决策树数量。
- criterion='entropy'：我们选择使用熵（entropy）作为分裂标准。
- random_state=0：设置一个随机种子以确保结果的可重复性。

这些是代码中的关键步骤和方法，希望这些解释有助于你理解它们的作用和使用。接下来，我们将继续执行模型评估和结果可视化的步骤。

## 步骤5：模型预测

我们使用训练好的SVM模型对测试集进行预测。

```python
# 在测试集上进行预测
y_pred = classifier.predict(X_test)
```
## 步骤6：评估模型性能

在这一步中，我们评估了SVM模型的性能，了解模型的准确性和其他性能指标。我们使用了混淆矩阵和分类报告来完成这项任务。

### 6.1 混淆矩阵（Confusion Matrix）

```python
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)
```

- `confusion_matrix` 函数接受两个参数：真实的目标值 `y_test` 和模型预测的目标值 `y_pred`。
- 混淆矩阵是一个二维数组，它展示了模型的分类性能。它的四个元素分别是真正例（True Positives，TP）、真负例（True Negatives，TN）、假正例（False Positives，FP）和假负例（False Negatives，FN）的数量。这些元素用于计算精确度、召回率等性能指标。

### 6.2 分类报告（Classification Report）

```python
print("\n分类报告:")
print(classification_report(y_test, y_pred))
```

- `classification_report` 函数接受两个参数：真实的目标值 `y_test` 和模型预测的目标值 `y_pred`。
- 分类报告提供了关于模型性能的详细信息，包括精确度（Precision）、召回率（Recall）、F1分数（F1-score）等。它对每个类别进行了单独的评估，并提供了加权平均值（weighted average）。

这些方法和参数是评估分类模型性能时常用的工具，可以帮助我们理解模型的强项和弱项，以便进一步优化和改进模型。

```
混淆矩阵:
[[63  5]
 [ 4 28]]

分类报告:
              precision    recall  f1-score   support

           0       0.94      0.93      0.93        68
           1       0.85      0.88      0.86        32

    accuracy                           0.91       100
   macro avg       0.89      0.90      0.90       100
weighted avg       0.91      0.91      0.91       100
```

## 步骤7：可视化结果
```python
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['red', 'green']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=['red', 'green'][i], label=j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['red', 'green']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=['red', 'green'][i], label=j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

```
![](https://cos.ywenrou.cn/blog/images20231011101553.png)
![](https://cos.ywenrou.cn/blog/images20231011101536.png)

