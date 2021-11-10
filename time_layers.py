import numpy as np

class RNN:
    def __init__(self,Wx , Wh , b):
        self.params = [Wx , Wh , b]
        self.cache = None

    def forward(self , x , h_pre):
        Wx , Wh , b = self.params
        temp = np.dot(x , Wx)+np.dot(h_pre , Wh)+b
        h = np.tanh(temp)

        self.cache = (x , h_pre , h)
        return h