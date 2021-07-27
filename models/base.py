import torch
import torch.nn as nn
from abc import abstractmethod

class BaseVAE(nn.Module):

    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError
    
    def decode(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, input):
        pass

    
