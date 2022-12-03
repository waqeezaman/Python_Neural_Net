from Layer import *
import numpy as np
class ActivationLayer(Layer):

    def __init__(self,activation,activation_deriv):
        super().__init__()
        ## set activation function and its derivative
        self.Activation=lambda x:activation(x)
        self.Activation_Deriv=activation_deriv

    def FeedForward(self, inputs):
        ## apply activation function to inputs
        self.Inputs=inputs
        return self.Activation(inputs)
    
    def BackPropogate(self, error_output_deriv, lr=0.01):
        ## return error-output-deriv with activation deriv applied
       ## print(self.Activation_Deriv(error_output_deriv))
        return np.multiply(self.Activation_Deriv(self.Inputs),error_output_deriv)

    
