import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)

    def forward(self, input_tensor):
        exp = np.exp(input_tensor)
        return exp / np.sum(exp, axis=0)

    def backward(self, error_tensor):
        # TODO:
        # avoid loops 
        pass