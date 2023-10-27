# -*- coding: utf-8 -*-
# @Author  : xuxiaoyang
# @Time    : 2023/10/15 14:05
# @Describe:
# 导入所需的库
import matplotlib.pyplot as plt
import matplotlib
from  sklearn.cluster import KMeans
from sklearn.datasets import load_iris

#设置 matplotlib rc配置文件
matplotlib.rcParams['font.sans-serif'] = [u'SimHei'] # 用来设置字体样式以正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False # 设置为 Fasle 来解决负号的乱码问题

# 加载鸢尾花数据集
# 数据的特征分别是 sepal length(花萼长度)、sepal width(花萼宽度)、petal length（花瓣长度）、petal width（花瓣宽度）
iris = load_iris()
X = iris.data[:, :2]  # 通过花萼的两个特征（长度和宽度）来聚类
k = 3  # 假设聚类为 3 类，默认分为 8 个 簇
# 构建算法模型
km = KMeans(n_clusters=k) # n_clusters参数表示分成几个簇（此处k=3）
km.fit(X)

# 获取聚类后样本所属簇的对应编号（label_pred）
label_pred = km.labels_  # labels_属性表示每个点的分簇号，会得到一个关于簇编号的数组
centroids = km.cluster_centers_  #cluster_center 属性用来获取簇的质心点，得到一个关于质心的二维数组，形如[[x1,y1],[x2,y2],[x3,x3]]

# 未聚类前的数据分布图
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.title("未聚类之前")
# wspace 两个子图之间保留的空间宽度
plt.subplots_adjust(wspace=0.5) # subplots_adjust（）用于调整边距和子图间距
# 聚类后的分布图
plt.subplot(122)
# c：表示颜色和色彩序列，此处与 cmap 颜色映射一起使用（cool是颜色映射值）s表示散点的的大小，marker表示标记样式（散点样式）
plt.scatter(X[:, 0], X[:, 1], c=label_pred, s=50, cmap='cool')
# 绘制质心点
plt.scatter(centroids[:,0],centroids[:,1],c='red',marker='o',s=100)
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.title("K-Means算法聚类结果")
plt.show()