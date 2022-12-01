from NeuralNetwork import *
from DenseLayer import *
import numpy as np
from random import randint


def linearfunc(x):
    return 3*x +9


a=NeuralNetwork([DenseLayer(1,1)],0.03)

##for i in range(0,200):
inp=np.array([1]).reshape(1)
out=np.array(1).reshape(1)
for i in range(10000):
    x=np.array(randint(-5,5))
    y=linearfunc(x)
    a.Train(x,y)

print(a.Query(np.array(3)))

