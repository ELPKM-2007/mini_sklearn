import numpy as np
class mini_GDLinearRegression:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.coef_=None
        self.intercept_=None
    def fit(self,x_train,y_train):
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        #一维梯度下降
        if x_train.ndim==1 or (x_train.ndim==2 and x_train.shape[1]==1):
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
        #多维梯度下降
        else:
            n_samples,n_features=x_train.shape[0],x_train.shape[1]
            w=np.zeros(n_features)
            b=0.0
            for e in range(self.epochs):
                y_pred=x_train @ w +b
                error=y_pred-y_train
                # 计算梯度 (按照 MSE 损失函数 L = (1/n) * sum(error^2))
                dw = (2 / n_samples) * (x_train.T @ error)
                db = (2 / n_samples) * np.sum(error)
                w-=self.lr*dw
                b-=self.lr*db
            self.coef_=w
            self.intercept_=b
            

    def predict(self,x_test):
        if self.intercept_ is None or self.coef_ is None:
            raise ValueError("模型未训练")
        x_test=np.array(x_test)
        if x_test.ndim == 1:
            x_test=x_test.reshape(-1,1)
        return x_test @ self.coef_+self.intercept_
    
if __name__=='__main__':
    x_train=[1,2,3,4,5,6]
    y_train=[4,5,6,7,8,9]
    estimator=mini_GDLinearRegression()
    estimator.fit(x_train,y_train)
    print(estimator.coef_)
    print(estimator.intercept_)
