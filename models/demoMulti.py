import torch
from torch import nn

##input seq length is 1000
##pred length is 1002

class demoMulti(nn.Module):
    def __init__(self):
        super(demoMulti, self).__init__()
        
        self.model_one = nn.Sequential(
                nn.Conv1d(4,640, 8, padding = 3),
                nn.BatchNorm1d(640),
                nn.ReLU(),
                nn.Conv1d(640,640, 8, padding = 4),
                nn.BatchNorm1d(640),
                nn.ReLU())
        
        self.model_two = nn.Sequential(
             
                nn.Conv1d(640,4008, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(4008),
                nn.ReLU(),
                nn.Conv1d(4008,4008, 8, dilation=4, padding= 14))
                



    def forward(self, x):
        layer_one = self.model_one(x)
        layer_two = self.model_two(layer_one) 
        final_out = torch.reshape(layer_two , (layer_two.shape[0], 1002, 4, 1000))
     
        return final_out