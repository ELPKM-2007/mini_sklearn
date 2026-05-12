from collections import Counter
import numpy as np
class mini_Kneighbors:
    #实际上就是把一个数据和每个数据进行对比，计算距离，选取最近的k个数据，然后让这k个数据进行投票
    def __init__(self,n_neighbors=5):
        self.k = n_neighbors
        pass
    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
    def _get_euclidean_distance(self,x1,x2):
        return np.sum((x1-x2)**2)**0.5
    def _get_manhattan_distance(self,x1,x2):
        return np.sum(np.abs(x1 - x2))
    def predict(self,x_test,way='euclidean'):
        x_test=np.asarray(x_test)
        y_pred=[]
        if way.lower()=='euclidean':
            distance_func=self._get_euclidean_distance
        elif way.lower()=='manhattan':
            distance_func=self._get_manhattan_distance
        else:
            raise ValueError('way只能是Euclidean或Manhattan')
        for x in x_test:
            distances=[]
            for i,x_train in enumerate(self.x_train):
                distance=distance_func(x_train,x)
                #还要有这个x_train对应的y_train
                y_train=self.y_train[i]
                distances.append((distance,y_train))
            #此时，我们有了当前x到所有x_train的距离
            #valid_ytrain存储参与投票的y_train
            valid_ytrain=[y_train for distance,y_train in sorted(distances)[:self.k]]
            #count存储(人:票数)
            count=Counter(valid_ytrain)
            y_test=count.most_common(1)[0][0]
            y_pred.append(y_test)
        return y_pred