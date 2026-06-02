import numpy as np


class mini_LinerRegression:
    def __init__(self):
        self.coef_=None
        self.intercept_=None
    def fit(self,x_train,y_train):
        lenx=len(x_train)
        leny=len(y_train)
        if lenx!=leny:
            raise ValueError('特征与目标集不等长')
        x_train=np.array(x_train)
        #x样式[[x1],[x2],[x3]...]或者[x1,x2,x3...]
        y_train=np.array(y_train)
        #y样式[y1,y2,y3...]
        if x_train.ndim==1 or (x_train.ndim==2 and x_train.shape[1]==1):
            x_train = x_train.ravel()
            x_mean=np.mean(x_train)
            y_mean=np.mean(y_train)
            numerator=0
            denominator=0

            #用numpy优化一下速度
            x_gap=x_train-x_mean#x_gap是个一维数组
            y_gap=y_train-y_mean#同上
            numerator=np.sum(x_gap*y_gap)
            denominator=np.sum((x_gap)**2)
            self.coef_=numerator/denominator
            self.intercept_=y_mean-self.coef_*x_mean
        else:
            #添加一列1作为截距列
            x_train=np.c_[np.ones((x_train.shape[0],1)),x_train]
            w=np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
            self.intercept_=w[0]
            self.coef_=w[1:]



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
    estimator=mini_LinerRegression()
    estimator.fit(x_train,y_train)
    print(estimator.coef_)
    print(estimator.intercept_)