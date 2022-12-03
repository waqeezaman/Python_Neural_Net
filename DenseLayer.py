from Layer import *
import numpy as np

class DenseLayer(Layer):
    
    def __init__(self,inputsize,outputsize):
        super().__init__()
        ## initialise weights and biases randomly
        self.Weights=np.random.randn(outputsize,inputsize)
        self.Bias=np.random.randn(outputsize,1)

    def FeedForward(self,inputs):
        self.Inputs=inputs
      
        ## matrix multiplication of weights and input vector plus the bias
        self.Outputs=np.dot(self.Weights,self.Inputs)+self.Bias
        return self.Outputs
    

    def BackPropogate(self,error_output_deriv, lr=0.001):
        
        ## derivative of weights to error is the inputs 
        weights_deriv=np.dot(error_output_deriv,np.transpose(self.Inputs))

        ## decrease weights and bias in direction opposite to gradient, 
        ## decrease is proportional to gradient 
        ## decrease is also restricted by learning rate 

        self.Weights-=(weights_deriv*lr)

        ## derivative of bias to error is just 1 
        self.Bias-=(error_output_deriv*lr)

        return np.dot(self.Weights.T,error_output_deriv)

    def Output(self):
        print("-----------------------------")
        print("WEIGHTS")
        print(self.Weights)
        print("BIAS")
        print(self.Bias)
        print("-----------------------------")