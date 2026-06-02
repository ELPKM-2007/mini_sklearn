from collections import Counter, defaultdict
from math import log2, sqrt
import numpy as np
from mini_decision_tree_classifier import mini_DecisionTreeClassifier

class mini_RandomForestClassifier:
    """
    n_estimators:对于每份数据,使用多少棵决策树.(随机森林中的树是固定的)
    max_features:决策树中寻找分裂点可能所需,增加随机性
    min_samples_split,max_depth,criterion:决策树所需参数
    bootstrap:每次给决策树训练样本时是否改用"随机抽一部分"
    """
    def __init__(self,n_estimators,bootstrap=True,min_samples_split=2,max_depth=3,criterion='gini',max_features=None):
        self.n_estimators=n_estimators
        self.max_features=max_features
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.criterion=criterion
        self.bootstrap=bootstrap
        self.forest=[]

    """
    获得self.forest,内含n_estimators棵决策树
    若有bootstrap,每颗决策树的训练样本都大概不同,同时决策树的每个分裂点的决策特征也有随机性,大大提高随机森林的随机性
    """
    def fit(self,x_train,y_train):
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        n_samples,n_features=x_train.shape
        max_features=self._get_max_features(n_features)
        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample_idx=np.random.choice(n_samples,n_samples)
            else:
                sample_idx=range(n_samples)
            x_samples=x_train[sample_idx]
            y_samples=y_train[sample_idx]
            tree=mini_DecisionTreeClassifier(self.min_samples_split,self.max_depth,self.criterion,max_features)
            tree.fit(x_samples,y_samples)
            self.forest.append(tree)
    



    """
    根据随机森林的self.max_features形式,返回一个对应的max_features
    """
    def _get_max_features(self,n_features):
        if self.max_features is None or self.max_features=='sqrt':
            return int(sqrt(n_features))
        elif self.max_features=='log2':
            return int(log2(n_features))
        elif isinstance(self.max_features,int):
            return min(self.max_features,n_features)
        elif isinstance(self.max_features,float):
            if self.max_features>1:
                raise ValueError("浮点型max_features不可大于1")
            return int(max(1,self.max_features*n_features))
        else:
            raise ValueError("Unsupported max_features")


    """
    预测某一行的结果
    直接用上所有树,投票决定这一行的结果
    """
    def _predict_one(self,x):
        result=[]
        for tree in self.forest:
            #每个树得到一个结果,最多的是最终结果
            result.append(tree.predict([x])[0])
        count=Counter(result)
        res=count.most_common(1)[0][0]
        return res

    """
    借助封装方法_predict_one,预测每一行的结果
    """

    def predict(self,x_test):
        x_test=np.array(x_test)
        if x_test.ndim==1 or x_test.shape[1]==1:
            x_test=x_test.ravel()
        y_pred=[]
        for x in x_test:
            y_pred.append(self._predict_one(x))
        return np.array(y_pred)




if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier as SKRandomForest

    # 生成一个稍复杂的二分类数据集
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=7, n_redundant=2,
        n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. 训练自定义的随机森林
    my_rf = mini_RandomForestClassifier(
        n_estimators=20,
        max_depth=5,
        min_samples_split=5,
        criterion='gini',
        max_features='sqrt',
        bootstrap=True
    )
    my_rf.fit(X_train, y_train)
    my_pred = my_rf.predict(X_test)
    my_acc = accuracy_score(y_test, my_pred)

    # 2. 训练 sklearn 的随机森林（相同参数）
    sk_rf = SKRandomForest(
        n_estimators=20,
        max_depth=5,
        min_samples_split=5,
        criterion='gini',
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    sk_rf.fit(X_train, y_train)
    sk_pred = sk_rf.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_pred)

    # 3. 输出对比结果
    print("=" * 50)
    print("随机森林测试结果对比")
    print("=" * 50)
    print(f"mini_RandomForestClassifier 准确率: {my_acc:.4f}")
    print(f"sklearn 随机森林准确率:           {sk_acc:.4f}")
    print("")
    print("预测结果是否一致？", np.array_equal(my_pred, sk_pred))
    print("（注意：由于随机森林内部的随机性，即使设置相同种子也可能不完全一致，但准确率应相近）")