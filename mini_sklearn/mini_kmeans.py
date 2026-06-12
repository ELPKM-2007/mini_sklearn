import numpy as np

class mini_Kmeans:
    """
    n_clusters:要把样本分为多少簇
    max_iters:最大迭代次数
    random_state:numpy随机种子
    """
    def __init__(self,n_clusters=3,max_iters=100,random_state=1):
        self.k=n_clusters
        self.max_iters=max_iters
        self.random_state=random_state
    def fit(self,x):
        x=np.array(x)
        n_samples,n_features=x.shape
        np.random.seed(self.random_state)
        #随机初始化k个中心,存储到self.centroids种
        indices=np.random.choice(n_samples,self.k,replace=True)
        self.centroids=x[indices].copy()

        for _ in range(self.max_iters):
            #计算每个样本到每个中心的距离,取最近的
            distances=np.linalg.norm(x[:,np.newaxis,:]-self.centroids,axis=2)
            self.labels=np.argmin(distances,axis=1)

            #用每个簇均值作为中心
            new_centroids=np.array(x[self.labels==i].mean(axis=0) for i in range(self.k))

            #收敛
            if np.allclose(self.centroids,new_centroids):
                break
            self.centroids=new_centroids
    def predict(self,x):
        distances=np.linalg.norm(x[:,np.newaxis,:]-self.centroids,axis=2)
        return np.argmin(distances,axis=1)



"""
曾梦想仗剑走天涯，看一看世界的繁华。
"""