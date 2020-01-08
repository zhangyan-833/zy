import numpy as np
from sigmoid import Sigmoid
from affine import Affine


def acc(w1,b1,w2,b2,x,t):
    accuracy_cnt=0
    affine1 = Affine(w1, b1)
    affine2 = Affine(w2, b2)
    sigmoid = Sigmoid()
    x1 = affine1.forward(x)
    y1 = sigmoid.forward(x1)
    x2 = affine2.forward(y1)
    for i in range(len(t)):
        if abs(x2[i][0]-t[i][0])<0.5:
            accuracy_cnt +=1
        #print(x2[i][0],t[i][0])
    print(accuracy_cnt/64)




