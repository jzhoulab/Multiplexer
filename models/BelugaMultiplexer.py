import torch
from torch import nn



class BelugaMultiplexer(nn.Module):
    def __init__(self):
        super(BelugaMultiplexer, self).__init__()
        
        self.model_one = nn.Sequential(
                nn.Conv1d(5,640, 8, padding = 3),
                nn.BatchNorm1d(640),
                nn.ReLU(),
                nn.Conv1d(640,640, 8, padding = 4),
                nn.BatchNorm1d(640),
                nn.ReLU())
        
        self.model_two = nn.Sequential(
             
                nn.Conv1d(640,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU())
                
        self.model_three = nn.Sequential(
            
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU())
        
        
        self.model_four = nn.Sequential(
                nn.Conv1d(1280, 1280, 8, dilation=64, padding= 224),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=64, padding= 224),
                nn.BatchNorm1d(1280),
                nn.ReLU())
        
        
        self.model_five = nn.Sequential(
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=16, padding= 56),
                nn.BatchNorm1d(1280),
                nn.ReLU())
        
        
        self.model_six = nn.Sequential(
                nn.Conv1d(1280,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU(),
                nn.Conv1d(1280,1280, 8, dilation=4, padding= 14),
                nn.BatchNorm1d(1280),
                nn.ReLU())
            
        self.model_final = nn.Sequential(               
                nn.Conv1d(1280,8008, 1, dilation=1, padding=0),
                nn.BatchNorm1d(8008),
                nn.ReLU(), 
                nn.Conv1d(8008,8008, 1, dilation=1, padding=0))


    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoding = torch.cat((torch.arange(0,1,0.001), torch.arange(1,0, -0.001))).unsqueeze(0).to(device)
        encoding = encoding.repeat(x.shape[0], 1, 1)
        x = torch.cat((x.to(device), encoding), dim = 1)
        layer_one = self.model_one(x)
        layer_two = self.model_two(layer_one)
        layer_three = self.model_three(layer_two)
        layer_four = self.model_four(layer_three)
        layer_five = self.model_five(layer_four)
        layer_six= self.model_six(layer_three + layer_five)
        final_out = self.model_final(layer_two + layer_six)        
        final_out = torch.reshape(final_out , (final_out.shape[0], 2002, 4, 2000))
     
        return final_out
    
    
