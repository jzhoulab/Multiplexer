import torch
from torch import nn
import numpy as np

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
    
class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        return self.model(x)
    
    
    
def log_fold(alt, ref):
    """
    Returns the log fold of a,b
    
    returns log(((alt+1e-6) * (1-ref+1e-6)) /((1-alt+1e-6) * (ref+1e-6)) 
    """
    e = 10**(-6)
    top = (alt + e)*(1 - ref + e)
    bot = (1 - alt + e) * (ref + e)
    return np.log(top/bot)


def batch_correlation(a,b):
    """returns corr(a,b) by calculating
    in batches
    
    
    
    """
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    a_var = 0
    b_var = 0
    covariance = 0
    
    batch_size = np.floor(len(a)**(1/2))
    for i in range(math.floor(len(a)/batch_size) + 1):
        
       
        batch_a = a[int(batch_size*i):int((i+1)*batch_size)]
        batch_b = b[int(batch_size*i):int((i+1)*batch_size)]
        
        if len(batch_a) == 0:
            return covariance/ np.sqrt(a_var*b_var)            
        
        batch_ab_size = len(batch_a)
        
        a_batch_mean = np.mean(batch_a)
        b_batch_mean = np.mean(batch_b)
        
        a_var += np.sum((batch_a - a_batch_mean)**(2)) + batch_ab_size*(a_batch_mean - a_mean)**(2)
        b_var += np.sum((batch_b - b_batch_mean)**(2)) + batch_ab_size*(b_batch_mean - b_mean)**(2)
        
        covariance += np.sum((batch_a - a_batch_mean)*(batch_b - b_batch_mean)) + batch_ab_size*(a_batch_mean - a_mean)*(b_batch_mean - b_mean)
        

    return covariance/ np.sqrt(a_var*b_var)


def encode_seq(seq):
    """
    returns an encoded sequence 
    
    Args:
        seq: 2000bp sequence
    
    Returns:
        4 x 2000 np.array

    """

    #encode the sequence
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}
    

    #each column is the encoding for each nucleotide in the original seq
    seq_encoded = np.zeros((4, len(seq)))
    for i in range(len(seq)):
        #this implements the encoding
        seq_encoded[:,i] = mydict[seq[i]]


        
    return torch.from_numpy(seq_encoded)


def mutate_seq(seq):
    """
    returns an encoded sequence and every possible mutation
    
    Args:
        seq: 2000bp sequence (encoded)
    
    Returns:
        6000bp mutations encoded to a 6000 x 4 x 2000 np.array

    """

    #encode the sequence
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}
    


    seq_encoded = encode_seq(seq)
    seq_encoded_tile = np.tile(seq_encoded, (8000, 1, 1)) #changes to 8000, 4, 2000
    
    for j in range(len(seq)):
        #for each element in the original sequence, the next three "layers" of 
        #seq_encoded_tile is a mutation
        i = j*4
        seq_encoded_tile[i, :, j] = mydict["A"]
        seq_encoded_tile[i + 1, :, j] = mydict["G"]
        seq_encoded_tile[i + 2, :, j] = mydict["C"]
        seq_encoded_tile[i + 3, :, j] = mydict["T"]
        
        
    return torch.from_numpy(seq_encoded_tile).float()