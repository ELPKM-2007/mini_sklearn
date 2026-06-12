from collections import Counter
from math import inf
import pandas as pd
import numpy as np
from math import log2

class mini_DecisionTreeClassifier:
    class _Node:
        """
        节点(决策树):
        feature:当前节点分裂的决策特征
        threshold:决策特征的阈值,决定当前特征集决策到左子树还是右子树
        left/right:左子树/右子树
        depth:当前节点深度,初始为0,最大允许为max_depth
        value:如果是叶子节点,则表示决策出来的特征
        """
        def __init__(self,feature=None, threshold=None, left=None, right=None, depth=None, value=None):
            self.feature=feature
            self.threshold=threshold
            self.left=left
            self.right=right
            self.depth=depth
            self.value=value
    """
    criterion:不纯度,评判数据纯度的参考:
        gini类似于熵越小越好.
        entropy信息增益(率)表示单独这个特征让总熵下降了多少,越大越好.这里函数计算的是条件熵,终究同gini,越小越好.
    min_samples_split:节点允许分裂的最小样本数(一般设置为2或者更大)
    max_depth:决策树最大深度
    max_features:针对随机森林实现,实现每个决策点只采用部分特征
    """
    def __init__(self, min_samples_split=2, max_depth=3, criterion='gini', max_features=None):
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_tree=None
        self.max_features=max_features
    
    def fit(self,x_train,y_train,sample_weight=None):
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        n=x_train.shape[1]
        if self.max_features is None:
            self.max_features_=n
        elif isinstance(self.max_features,int):
            self.max_features_=min(self.max_features,n)
        else:
            raise ValueError("max_features must be int or None")
        self.decision_tree=self._build_tree(x_train,y_train,sample_weight)

    def _predict_one(self,node,x):
        if node.value is not None:
            return node.value
        if x[node.feature]<=node.threshold:
            return self._predict_one(node.left,x)
        else:
            return self._predict_one(node.right,x)
    def predict(self,x_test):
        root=self.decision_tree
        y_pred=[]
        for x in x_test:
            y_pred.append(self._predict_one(root,x))
        return np.array(y_pred)
    """
    #计算标签数组y的基尼值gini
    如果某个样本的权重很大，它在计算 count[lab] 时贡献就大，从而影响基尼系数。 决策树会倾向于选择能使加权基尼下降最多的分裂点。
    """
    def _gini(self,y,sample_weight=None):
        # y=np.array(y)
        if y.ndim==1 or y.shape[1]==1:
            y=y.ravel()
        else:
            raise ValueError("一维列表才能计算基尼值")
        res=1
        if sample_weight is None:
            n=len(y)
            count=Counter(y)
            for v in count.values():#v就是出现次数
                res-=(v/n)**2
        else:
            total=sample_weight.sum()
            count={}
            for i,lab in enumerate(y):
                count[lab]=count.get(lab,0)+sample_weight[i]#它的权重越大,那么它的"次数就越多"
            for v in count.values():
                res-=(v/total)**2
        return res
    
    #计算信息增益的条件熵,条件熵越小越好,和上边的gini一样
    def _entropy(self,y,sample_weight=None):
        # y=np.array(y)
        if y.ndim==1 or y.shape[1]==1:
            y=y.ravel()
        else:
            raise ValueError("一维列表才能计算信息增益值")
        res=0
        if sample_weight is None:
            n=len(y)
            count=Counter(y)
            for v in count.values():
                res-=(v/n)*log2(v/n)
        else:
            total=sample_weight.sum()
            count={}
            for i,lab in enumerate(y):
                count[lab]=count.get(lab,0)+sample_weight[i]
            for v in count.values():
                res-=(v/total)*log2(v/total)
        return res

    """
    对于特征集x与标签集y
    返回x的最好的特征及其分裂阈值
    用于寻找决策点特征(以及阈值)
    可以利用self.max_features实现随机选择max_featrues个特征来寻找最优分裂,提高了随机性.若仅研究单棵决策树可以视为None
    """
    def _best_split(self,x,y,sample_weight=None):
        # y=np.array(y)
        # x=np.array(x)
        if self.criterion=='gini':
            func=self._gini
        elif self.criterion=="entropy":
            func=self._entropy
        else:
            raise ValueError("")
        best_feature, best_threshold, best_impurity=None,None,inf
        m,n=x.shape
        if self.max_features_<n:
            features_idxes=np.random.choice(n,self.max_features_,replace=False)#False表示不放回抽取,可以防止重复抽取某个特征
        else:
            features_idxes=range(n)
        if sample_weight is None:
            for j in features_idxes:
                cur_x=x[:,j]#curx是当前要处理的列
                l=len(cur_x)
                sort_idx=np.argsort(cur_x)
                cur_x=cur_x[sort_idx]
                cur_y=y[sort_idx]#直接建立新的数组
                impurityval=inf#这一列的最优impurity值
                threshold=None#最优impurity值对应的阈值
                for i in range(l-1):
                    if cur_x[i]==cur_x[i+1]:
                        continue
                    impurityleft=func(cur_y[:i+1])*(i+1)/l
                    impurityright=func(cur_y[i+1:])*(l-i-1)/l
                    curimpurityval=impurityleft+impurityright#此阈值的加权impurity系数
                    if curimpurityval<impurityval:
                        impurityval=curimpurityval
                        threshold=(cur_x[i]+cur_x[i+1])/2
                if impurityval<best_impurity:
                    best_impurity=impurityval
                    best_feature=j
                    best_threshold=threshold
            return (best_feature, best_threshold, best_impurity)#最好的特征,阈值,对应的impurity系数值(但是没啥用)
        else:
            """
            决策树会计算每个可能分裂点的加权不纯度，选择能最小化加权不纯度的特征和阈值。 这就实现了“更关注权重大样本”的目标：因为权重大样本对不纯度的贡献更大，决策树会努力把它们分对。
            """
            total=sample_weight.sum()
            for j in features_idxes:
                cur_x=x[:,j]
                sort_idx=np.argsort(cur_x)
                cur_x=cur_x[sort_idx]
                cur_y=y[sort_idx]
                cur_w=sample_weight[sort_idx]
                impurityval=inf
                threshold=None
                left_weight=0.0
                l=len(cur_x)
                for i in range(l-1):
                    left_weight+=cur_w[i]
                    if cur_x[i]==cur_x[i+1]:
                        continue
                    right_weight=total-left_weight
                    impurityleft=func(cur_y[:i+1],cur_w[:i+1])
                    impurityright=func(cur_y[i+1:],cur_w[i+1:])
                    cur_impurity=(left_weight/total)*impurityleft+(right_weight/total)*impurityright#权重核心体现在这里
                    if cur_impurity < impurityval:
                        impurityval=cur_impurity
                        threshold=(cur_x[i]+cur_x[i+1])/2
                if impurityval<best_impurity:
                    best_impurity=impurityval
                    best_feature=j
                    best_threshold=threshold
            return (best_feature,best_threshold,best_impurity)


    """
    递归构建决策树
    仅叶子节点有value值表示预测的标签
    self.max_features已经体现在了每次决策best_split上
    """
    def _build_tree(self,x,y,sample_weight=None,depth=0):
        if sample_weight is None:
            # x=np.array(x)
            # y=np.array(y)
            #递归终止条件:达到决策树最大深度,y只有一种标签,y的标签个数小于最小分裂
            if depth==self.max_depth or len(np.unique(y)) == 1 or len(y)<self.min_samples_split:
                node=self._Node(depth=depth)
                node.value=np.bincount(y).argmax()
                return node

            best_feature, best_threshold, best_impurity=self._best_split(x,y)
            if best_feature is None:
                node=self._Node(depth=depth)
                node.value=np.bincount(y).argmax()
                return node
            #mask:Bool筛选决策到左右子树的部分
            left_mask=x[:,best_feature]<=best_threshold
            right_mask=~left_mask
            #也就是x[:,best_feature]>best_threshold


            #防御,但大概不会走这里
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                node=self._Node(depth=depth)
                node.value=np.bincount(y).argmax()
                return node
            #按照最优特征与阈值分裂x,y.递归构建左右子树
            node=self._Node(feature=best_feature,threshold=best_threshold,depth=depth)
            node.left=self._build_tree(x=x[left_mask],y=y[left_mask],depth=depth+1)
            node.right=self._build_tree(x=x[right_mask],y=y[right_mask],depth=depth+1)
            return node
        else:
            if depth==self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
                node=self._Node(depth=depth)
                classes=np.unique(y)
                best_class=max(classes,key=lambda c:sample_weight[y==c].sum())
                node.value=best_class
                return node
            best_feature,best_threshold,best_impurity=self._best_split(x,y,sample_weight)
            if best_feature is None:
                node=self._Node(depth=depth)
                classes=np.unique(y)
                best_class=max(classes,key=lambda c:sample_weight[y==c].sum())
                node.value=best_class
                return node
            left_mask=x[:,best_feature]<=best_threshold
            right_mask=~left_mask
            if np.sum(left_mask)==0 or np.sum(right_mask)==0:
                node=self._Node(depth=depth)
                classes=np.unique(y)
                best_class=max(classes,key=lambda c:sample_weight[y==c].sum())
                node.value=best_class
                return node
            node=self._Node(feature=best_feature,threshold=best_threshold,depth=depth)
            node.left=self._build_tree(x[left_mask],y[left_mask],sample_weight[left_mask],depth+1)
            node.right=self._build_tree(x[right_mask],y[right_mask],sample_weight[right_mask],depth+1)
            return node

        




"""
以下是测试代码
"""
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, 
                            n_redundant=1, n_classes=2, random_state=42)
    # 1. 使用 iris 数据集，取前两类做二分类
    X = X[y != 2]   # 只保留类别 0 和 1
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

  # 测试1：不传权重，应该和原来结果一致
    my_clf = mini_DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion='gini')
    my_clf.fit(X_train, y_train)
    my_pred = my_clf.predict(X_test)
    print(accuracy_score(y_test, my_pred))

    # 测试2：传入随机权重，不会报错即可
    w = np.random.rand(len(y_train))
    my_clf.fit(X_train, y_train, sample_weight=w)