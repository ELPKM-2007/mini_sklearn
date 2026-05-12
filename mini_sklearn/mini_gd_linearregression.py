import numpy as np
from mini_change_ndim_1 import change#把一维数组转为[1,2,3...]
class mini_GDLinearRegression:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.coef_=None
        self.intercept_=None
    def fit(self,x_train,y_train):
        try:
            x_train=change(x_train)
            y_train=change(y_train)
        except:
            raise ValueError("格式仅支持一维")
        n=len(x_train)
        k=0.0
        b=0.0
        for e in range(self.epochs):
            y_pred=k*x_train+b
            error=y_pred-y_train
            #计算梯度
            grad_k = (2 / n) * np.sum(x_train * error)
            grad_b = (2 / n) * np.sum(error)
            #改变参数值
            k-=grad_k*self.lr
            b-=grad_b*self.lr
        self.coef_ = k
        self.intercept_ = b
            

    def predict(self,x_test):
        x_test=change(x_test)
        return self.coef_*x_test+self.intercept_
    
if __name__=='__main__':
    x_train=[1,2,3,4,5,6]
    y_train=[4,5,6,7,8,9]
    estimator=mini_GDLinearRegression()
    estimator.fit(x_train,y_train)
    print(estimator.coef_)
    print(estimator.intercept_)
