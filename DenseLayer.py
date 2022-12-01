from Layer import *
import numpy as np

class DenseLayer(Layer):
    
    def __init__(self,inputsize,outputsize):
        super().__init__()
        self.Weights=np.random.randn(outputsize,inputsize)
        self.Bias=np.random.randn(outputsize,1)

    def FeedForward(self,inputs):
        self.Inputs=inputs
      
        self.Outputs=np.dot(self.Weights,self.Inputs)+self.Bias
        return self.Outputs
    

    def BackPropogate(self,error_output_deriv, lr=0.001):
        weights_deriv=np.dot(error_output_deriv,np.transpose(self.Inputs))
        self.Weights-=weights_deriv*lr
      
        self.Bias-=error_output_deriv*lr
        return np.dot(self.Weights.T,error_output_deriv)