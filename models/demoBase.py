import torch
from torch import nn
#Beluga input shape is 1x4x1x2000
#multiplexer input shape is 1x5x2000

#Beluga Multiplexer is 1x2002x4x2000
#Beluga output is 1x2002

#dummyBase input 1x5x1000
#dummyBase output 1x1002

#dummyMultiplexer input is 1x5x1000
#dummyMultiplexer output is [1, 1002, 4, 1000]

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))
    
class demoBase(nn.Module):
    def __init__(self):
        super(demoBase, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(4,160,(1, 8)),
                nn.ReLU(),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(160,160,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(160,160,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(8480,1002)),
                nn.ReLU(),
                nn.Sigmoid()
            )
        


    def forward(self, x):
        x = x.unsqueeze(2)
        return self.model(x)