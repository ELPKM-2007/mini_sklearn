import numpy as np
from mini_change_ndim_1 import change#把一维数组转为[1,2,3...]


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
            pass#多元线性回归
    def predict(self,x_test):
        if self.intercept_ is None or self.coef_ is None:
            raise ValueError("模型未训练")
        x_test=np.array(x_test).ravel()
        return self.coef_*x_test+self.intercept_
