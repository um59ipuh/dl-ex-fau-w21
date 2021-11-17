import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)

    def forward(self, input_tensor):
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        pass
    