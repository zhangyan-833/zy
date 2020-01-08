import numpy as np
import matplotlib.pyplot as plt
from sigmoid import Sigmoid
from affine import Affine
from loss import MSE
from accurancy import acc
def init(input_size,hidden_size,output_size):
        params={}
        np.random.seed(31200)
        params['w1']=np.random.randn(input_size ,hidden_size )
        params['b1']=np.zeros(hidden_size)
        params['w2'] = np.random.randn( hidden_size, output_size )
        params['b2'] = np.zeros(output_size)
        return params['w1'],params['b1'],params['w2'],params['b2']

def main():

    train_data=np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],
                         [1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],
                         [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],
                         [3,0],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],
                         [4,0],[4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,7],
                         [5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[5,7],
                         [6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6],[6,7],
                         [7,0],[7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7]])
    label_data=np.array([[0],[1],[2],[3],[4],[5],[6],[7],
                         [1],[0],[3],[2],[5],[4],[7],[6],
                         [2],[3],[0],[1],[6],[7],[4],[5],
                         [3],[2],[1],[0],[7],[6],[5],[4],
                         [4],[5],[6],[7],[0],[1],[2],[3],
                         [5],[4],[7],[6],[1],[0],[3],[2],
                         [6],[7],[4],[5],[2],[3],[0],[1],
                         [7],[6],[5],[4],[3],[2],[1],[0]])
    learn_rate=0.15
    epoch=600000
    loss_list=[]
    w1,b1,w2,b2=init(input_size=2,hidden_size=16,output_size=1)
    #print(w1, w2, b1, b2)


    for i in range(epoch):
        affine1 = Affine(w1, b1)
        affine2 = Affine(w2, b2)
        sigmoid=Sigmoid()
        loss=MSE()
        x1=affine1.forward(train_data)
        y1=sigmoid.forward(x1)
        x2=affine2.forward(y1)
        ls=loss.mean_square_error(x2,label_data)
        print(ls)
        loss_list.append(ls)
        dout=loss.backward(x2,label_data)
        dx=affine2.backward(dout)
        w2=w2-learn_rate * affine2.dw
        b2=b2-learn_rate * affine2.db
        dy1=sigmoid.backward(dx)
        dx=affine1.backward(dy1)
        b1=b1-learn_rate * affine1.db
        w1=w1-learn_rate * affine1.dw
    #print(w1,w2,b1,b2)

    plt.plot(loss_list)
    plt.show()
    acc(w1,b1,w2,b2,train_data,label_data)
if __name__ == '__main__':
    main()