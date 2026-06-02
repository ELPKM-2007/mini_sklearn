# mini_sklearn
2026.6.2:删除工具函数mini_change_ndim_1
新增:逻辑回归，决策树，随机森林算法

## 模块说明

### mini_knn.py
KNN（K近邻）分类器，支持欧氏距离和曼哈顿距离。核心思想：计算测试样本与所有训练样本的距离，取最近的 k 个邻居投票决定分类结果。

### mini_linearregression.py
一元线性回归，使用最小二乘法（正规方程）直接求解最优参数。通过最小化残差平方和，一次性算出系数和截距。

### mini_gd_linearregression.py
一元线性回归的梯度下降实现。与正规方程不同，通过迭代更新参数逐步逼近最优解，可设置学习率和迭代轮数。

### mini_sklearn\mini_logicregression.py
逻辑回归,实际上就是换个损失函数的线性回归,借助sigmoid映射实现逻辑预测

### mini_sklearn\mini_decision_tree_classifier.py
决策树,支持参数min_samples_split, max_depth, criterion, max_features
大致实现逻辑:best_split返回传入的x,y对应的最优决策特征,阈值,以及对应criterion的熵值
build_tree中每个决策点都用一次best_split,接着根据所返回的特征和阈值将原数据集分为两部分,递归建树.
当某个节点所有标签相同或者达到条件min_samples_split, max_depth时,节点(叶子)便会用value值存储其预测结果

支持max_features,使得每次决策使用的特征集很可能不同,提高随机性

### mini_sklearn\mini_random_forest_classifier.py
随机森林,支持参数n_estimators,bootstrap,min_samples_split,max_depth,criterion,max_features
大致实现逻辑:fit以实现self.forest,每棵树的训练均使用不同的数据集(量相同内不同),同时训练过程也会因max_features产生很大的随机性
封装方法_predict_one实现单行预测,多棵决策树投票决定结果
predict借助predict_one方法实现数据集预测