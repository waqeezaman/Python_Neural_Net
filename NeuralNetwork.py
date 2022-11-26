import numpy as np
class NeuralNetwork:

    def __init__(self,layers,lr):
        self.Layers=layers
        self.CumulativeError=0
    
    def Query(self,inputs):
       return self.FeedForward(inputs)

    def Train(self,data,target):
        actual=self.Query(data)
        self.CumulativeError+=self.MeanSquaredError(actual,target)
        self.BackPropogate(self.MeanSquaredError_Deriv(actual,target))



    def FeedForward(self,inputs):
        nextInput=inputs ## sets the input of the first layer

        for i in range(1,len(self.Layers)):
             nextInput=self.Layers[i].FeedForward(nextInput) ## the output of one layer is the input of the next

        return self.Layers[len(self.Layers)-1].Outputs
    
    def BackPropogate(self,error_output_deriv):
            nextDeriv=self.Layers[len(self.Layers)-1].Backpropogate(error_output_deriv)
            for i in range(len(self.Layers)-2,0,-1):
                 nextDeriv=self.Layers[i].Backpropogate(nextDeriv)

    def MeanSquaredError(self,actual,target):
         return np.mean(np.power(actual-target,2))
    
    def MeanSquaredError_Deriv(actual,target):
         return 2/np.size(actual)*(actual-target)