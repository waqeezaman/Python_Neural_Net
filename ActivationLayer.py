from Layer import *
class ActivationLayer(Layer):

    def __init__(self,activation,activation_deriv):
        self.Activation=activation
        self.Activation_Deriv=activation_deriv

    def FeedForward(self, inputs):
        return self.Activation(inputs)
    
    def BackPropogate(self, error_output_deriv, lr):
        return self.Activation_Deriv(error_output_deriv)