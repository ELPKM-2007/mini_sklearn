import numpy as np
from mini_decision_tree_classifier import mini_DecisionTreeClassifier

class mini_GBDT_Regressor:
    def __init__(self,n_estimators=100,learning_rate=0.1,max_depth=3):
        self.n_estimators=n_estimators
        self.lr=learning_rate
        self.max_depth=max_depth
        self.trees=[]
    def fit(self,x_train,y_train):
        self.initital_pred=np.mean(y_train)#用y_mean作为初始预测值
        current_pred=np.full(x_train.shape[0],self.initital_pred)
        for _ in range(self.n_estimators):
            residual=y_train-current_pred

            tree=mini_DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(x_train,residual)
            update=tree.predict(x_train)

            current_pred+=self.lr*update
            self.trees.append(tree)
    def predict(self,x_test):
        y_pred=np.full(x_test.shape[0],self.initital_pred)
        for tree in self.trees:
            y_pred+=self.lr*tree.predict(x_test)
        return y_pred