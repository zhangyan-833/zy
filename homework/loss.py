import numpy as np

class MSE:
    def mean_square_error(self,y,t):
        n=y.shape[0]
        out =0.5*np.sum((y-t)**2)/n
        return out

    def backward(self,y,t):
        n = y.shape[0]
        dy=(y-t)/n
        return dy

