from NeuralNetwork import *
from DenseLayer import *
from ActivationLayer import *
import numpy as np
from random import randint
from ActivationFunctions import *


def linearfunc(x):
    return 1*x 


a=NeuralNetwork([DenseLayer(1,1),
                ##DenseLayer(3,1),
                ActivationLayer(RELU_array, RELU_Deriv_array)
               ]
                ,0.01)





for i in range(0,1000,1):
    x=np.array(randint(0,1))
    y=linearfunc(x)
    a.Train(x,y)

print("OUTPUT")
print(a.Query(np.array(3)))
print(a.Query(np.array(2)))
print(a.Query(np.array(1)))



a.OutputNetwork()


# b=np.array([[-3,2],[3.9,-2]])
# print(RELU_Deriv_array(b))


# layer=ActivationLayer(RELU_array,RELU_Deriv_array)
# inp=np.array([[-1],[-8]])
# print(layer.FeedForward(inp))