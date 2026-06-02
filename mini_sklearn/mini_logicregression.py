import numpy as np

class mini_LogicRegression:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.coef_=None
        self.intercept_=None
    def sigmoid(self,x):#把x映射到[0,1]中
        return 1/(1+np.exp(-x))
    def fit(self,x_train,y_train,debug=False):
        
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        n_samples,n_features=x_train.shape
        w=np.zeros((n_features))
        b=0
        for e in range(self.epochs):
            y_pred=x_train @ w + b
            y_pred=self.sigmoid(y_pred)
            #损失函数
            #Loss=-sum(y_train * log(y_pred) + (1-y_train) * log(1-y_pred))
            #y_pred中包含变量w和b
            #分别求偏导,就是梯度
            dw = (1 / n_samples) * (x_train.T @ (y_pred - y_train))
            db = (1 / n_samples) * np.sum(y_pred - y_train)
            w-=dw * self.lr
            b-=db * self.lr
            if debug and e % 100 == 0:
                loss = -np.mean(y_train * np.log(y_pred + 1e-8) + (1 - y_train) * np.log(1 - y_pred + 1e-8))
                print(f"Epoch {e}, loss: {loss:.6f}")
        self.intercept_=b
        self.coef_=w
    
    def score(self,x,y):
        y_pred=self.predict(x)
        return np.mean(y_pred == y)

    def predict_proba(self, x_test):
        x_test = np.array(x_test)
        linear = x_test @ self.coef_ + self.intercept_
        return self.sigmoid(linear)
    def predict(self,x_test,threshold=0.5):
        proba=self.predict_proba(x_test)
        return (proba>=threshold).astype(int)