class Layer:

    def __init__(self):
        self.Outputs=None
        self.Inputs=None
        pass
    def FeedForward(self,inputs):
        ## function to be implemented
        pass
    def BackPropogate(self,error_output_deriv,lr=0.01):
        ## function to be implemented 
        pass
    def Output(self):
        print()