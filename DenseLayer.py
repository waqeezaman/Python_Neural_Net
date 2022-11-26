from Layer import *
import numpy as np

class DenseLayer(Layer):
    
    def __init__(self,inputsize,outputsize):
        self.Weights=np.random.rand(outputsize,inputsize)
        self.Bias=np.random.rand(outputsize)

    def FeedForward(self,inputs):
        self.Inputs=inputs
        self.Outputs=np.dot(self.Weights,inputs)+self.Bias
        return self.Outputs
    
    def BackPropogate(self,error_output_deriv, lr):
        weights_deriv=np.dot(error_output_deriv,self.Inputs.T)
        self.Weights+=weights_deriv
        self.Bias+=error_output_deriv
        return np.dot(self.Weights.T,error_output_deriv)