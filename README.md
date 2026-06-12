# mini_sklearn

mini_knn.py
KNN（K近邻）分类器，支持欧氏距离和曼哈顿距离。核心思想：计算测试样本与所有训练样本的距离，取最近的 k 个邻居投票决定分类结果。

mini_linearregression.py
一元线性回归，使用最小二乘法（正规方程）直接求解最优参数。通过最小化残差平方和，一次性算出系数和截距。

mini_gd_linearregression.py
一元线性回归的梯度下降实现。与正规方程不同，通过迭代更新参数逐步逼近最优解，可设置学习率和迭代轮数。

mini_logicregression.py
逻辑回归，实际上就是换个损失函数的线性回归，借助sigmoid映射实现逻辑预测

mini_decision_tree_classifier.py
决策树,支持参数min_samples_split, max_depth, criterion, max_features 大致实现逻辑:best_split返回传入的x,y对应的最优决策特征,阈值,以及对应criterion的熵值 build_tree中每个决策点都用一次best_split,接着根据所返回的特征和阈值将原数据集分为两部分,递归建树. 当某个节点所有标签相同或者达到条件min_samples_split, max_depth时,节点(叶子)便会用value值存储其预测结果

支持max_features,使得每次决策使用的特征集很可能不同,提高随机性

mini_random_forest_classifier.py
随机森林,支持参数n_estimators,bootstrap,min_samples_split,max_depth,criterion,max_features 大致实现逻辑:fit以实现self.forest,每棵树的训练均使用不同的数据集(量相同内不同),同时训练过程也会因max_features产生很大的随机性 封装方法_predict_one实现单行预测，多棵决策树投票决定结果 predict借助predict_one方法实现数据集预测

mini_adaboost.py
注意：标签必须为 ±1 AdaBoost 自适应增强分类器，支持参数 n_estimators。 核心思想：使用决策树桩（max_depth=1）作为弱学习器，通过多轮迭代逐步提升模型能力。每轮训练后根据预测结果更新样本权重——被分错的样本获得更高权重，迫使下一轮弱学习器重点关注这些"难样本"。每轮还会计算该弱学习器的权重 alpha（取决于错误率），最终预测为所有弱学习器预测结果的加权投票。

mini_gbdt_regressor.py
GBDT 梯度提升决策树回归器，支持参数 n_estimators、learning_rate、max_depth。 核心思想：用所有样本标签的均值作为初始预测值，之后每轮计算当前预测值与真实值的残差，用一棵决策树去拟合残差，再按学习率将树的预测值叠加到当前结果上。经过多轮迭代，残差逐步减小，预测值逐步逼近真实值。可以理解为"每次修正一点点，一步步逼近目标"的加法模型。

mini_kmeans.py
K-Means 聚类算法（无监督学习），支持参数 n_clusters、max_iters、random_state。 核心思想：随机初始化 k 个聚类中心 → 将每个样本分配到距离最近的中心（形成 k 个簇）→ 用每个簇的均值更新中心 → 重复直到中心不再变化或达到最大迭代次数。该算法不需要标签，自动发现数据的内在分组结构。
