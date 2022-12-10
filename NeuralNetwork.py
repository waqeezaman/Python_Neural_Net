import numpy as np
class NeuralNetwork:

    def __init__(self,layers,lr,random_state=np.random.RandomState(seed=np.random.randint(0,9999999))):
        self.LearningRate=lr
        self.Layers=layers
        self.CumulativeError=0
        self.Random_State=random_state
    def Query(self,inputs):
       return self.FeedForward(inputs)

    def Train(self,data,target):
        actual=self.Query(data)
        ##print(actual-target)
        ##print("ERROR-DERIV")
        
        ##print(self.MeanSquaredError_Deriv(actual,target))
        ##self.CumulativeError+=self.MeanSquaredError(actual,target)
        self.BackPropogate(self.MeanSquaredError_Deriv(actual,target))



    def FeedForward(self,inputs):
        nextInput=inputs ## sets the input of the first layer

        for i in range(0,len(self.Layers),1):
            nextInput=self.Layers[i].FeedForward(nextInput) ## the output of one layer is the input of the next
            if(i==len(self.Layers)-1): 
                return nextInput##self.Layers[len(self.Layers)-1].FeedForward(nextInput)
    
    def BackPropogate(self,error_output_deriv):
            nextDeriv=error_output_deriv##self.Layers[len(self.Layers)-1].BackPropogate(error_output_deriv,lr=self.LearningRate)
            for i in range(len(self.Layers)-1,-1,-1):
                nextDeriv=self.Layers[i].BackPropogate(nextDeriv,lr=self.LearningRate)

    def MeanSquaredError(self,actual,target):
         return np.mean(np.power(actual-target,2))
    
    def MeanSquaredError_Deriv(self,actual,target):
         return 2*(actual-target)/np.size(actual)

    def OutputNetwork(self):
        for i in self.Layers:
            i.Output()
