import numpy as np
from mini_decision_tree_classifier import mini_DecisionTreeClassifier

class mini_AdaBoost:
    def __init__(self,n_estimators=50):
        self.n_estimators=n_estimators
        self.models=[]#存储所有弱学习器
        self.alphas=[]#存储每个学习器的权重
    def fit(self,x_train,y_train):
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        unique = np.unique(y_train)
        if not np.array_equal(unique, [-1, 1]):
            raise ValueError(f"AdaBoost:传入标签仅能为-1 或 1")
        n_samples,n_features=x_train.shape
        w=np.ones(n_samples)/n_samples#初始化w表示每个样本的权重
        for _ in range(self.n_estimators):
            #用加权样本训练树状
            stump=mini_DecisionTreeClassifier(max_depth=1)
            stump.fit(x_train,y_train,sample_weight=w)

            #预测,计算错误率
            pred=stump.predict(x_train)
            err=np.sum(w*(pred!=y_train))/np.sum(w)

            #计算模型权重
            alpha=0.5*np.log((1-err)/max(err,1e-10))

            #更新样本权重
            w=w*np.exp(-alpha*y_train*pred)
            w=w/np.sum(w)

            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self,x_test):
        pred=np.zeros(x_test.shape[0])
        for alpha,model in zip(self.alphas,self.models):
            pred+=alpha*model.predict(x_test)
        return np.sign(pred)
    

# test
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mini_decision_tree_classifier import mini_DecisionTreeClassifier


if __name__ == "__main__":
    # 1. 生成数据
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=8,
        n_redundant=2, n_classes=2, random_state=42
    )
    
    # 2. 转换标签为 ±1
    y_ada = np.where(y == 0, -1, 1)
    
    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ada, test_size=0.2, random_state=42
    )
    
    # 4. 训练你的 AdaBoost
    ada = mini_AdaBoost(n_estimators=50)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"你的 AdaBoost 准确率: {acc:.4f}")
    
    # 5. 对比 sklearn（如果有）
    try:
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        sk_ada = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1),
            n_estimators=50
        )
        sk_ada.fit(X_train, y_train)
        sk_pred = sk_ada.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_pred)
        print(f"sklearn AdaBoost 准确率: {sk_acc:.4f}")
    except ImportError:
        print("未安装 sklearn，跳过对比")
    # 在测试代码最后添加
    # 比较前5个预测结果是否完全相同
    print("前5个预测值对比：")
    print("  你的实现:", y_pred[:5])
    print("  sklearn :", sk_pred[:5])
    print("  是否完全一致:", np.all(y_pred == sk_pred))

    # 比较准确率的具体浮点值（不四舍五入）
    from sklearn.metrics import accuracy_score
    acc_you = accuracy_score(y_test, y_pred)
    acc_sk = accuracy_score(y_test, sk_pred)
    print(f"你的准确率精确值: {acc_you:.10f}")
    print(f"sklearn准确率精确值: {acc_sk:.10f}")