from NeuralNetwork import *
from DenseLayer import *
from ActivationLayer import *
import numpy as np
import math
from ActivationFunctions import *



def linearfunc(x):
    return 1*x 

random_state=np.random.RandomState(seed=0)
a=NeuralNetwork([DenseLayer(1,1,random_state=random_state),
                ActivationLayer(RELU_array, RELU_Deriv_array),
                DenseLayer(1,1,random_state=random_state),
                ActivationLayer(RELU_array, RELU_Deriv_array),
                DenseLayer(1,1,random_state=random_state)
               ]
                ,0.005)




for i in range(0,10000,1):
    x=np.array(np.random.RandomState(seed=i).random())
    y=math.sin(x)
    a.Train(x,y)

print("OUTPUT")
print(a.Query(np.array(0.25)))
print(a.Query(np.array(0.5)))
print(a.Query(np.array(1)))









