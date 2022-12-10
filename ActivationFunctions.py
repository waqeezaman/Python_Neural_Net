import numpy as np
def RELU(x):
        if x<0 :
            print("negative value")

            return 0.01*x

        else:
            return x


def RELU_Deriv(x):
        if x<0 :
            return 0.01
        else:
            return 1

def Sigmoid(x):
    pass

def Sigmoid_Deriv(x):
    pass


## vectorised functions to allow numpy arrays to be passed 
## applies function to numpy array element-wise

RELU_array = np.vectorize(RELU,otypes=[float])
RELU_Deriv_array = np.vectorize(RELU_Deriv,otypes=[float])

Sigmoid_array=np.vectorize(Sigmoid,otypes=[float])
Sigmoid_Deriv_array=np.vectorize(Sigmoid_Deriv,otypes=[float])