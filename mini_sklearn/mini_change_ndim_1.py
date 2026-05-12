import numpy as np

def change(x):
    #改成一维[1,2,3...]形式
    x=np.array(x)
    if x.ndim==2:
        if x.shape[0]==1 or x.shape[1]==1:
            x=x.ravel()
        else:
            raise ValueError("请传入(1,n)或(n,1)格式")
    elif x.ndim==1:
        x=x.ravel()
    else:
        raise ValueError("请传入(1,n)或(n,1)格式")
    return x
# class change_ndim:
#     def __init__(self,x):
#         self.x=x
#     def change(self,x):
#         #改成一维[1,2,3...]形式
#         x=np.array(self.x)
#         if x.ndim==2:
#             if x.shape[1]==1:
#                 x=x.ravel()
#             else:
#                 raise ValueError("请传入(1,n)或(n,1)格式")
#         elif x.ndim==1:
#             x=x.ravel()
#         else:
#             raise ValueError("请传入(1,n)或(n,1)格式")
#         return x