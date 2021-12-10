import numpy as np
from .Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)

    def forward(self, input_tensor):
        self.input = input_tensor
        batch = input_tensor.shape[0]
        output_arr = [input_tensor[img].flatten() for img in range(0, batch)]
        return np.array(output_arr)

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        output_arr = [error_tensor[img].reshape(self.input[img].shape) for img in range(0, batch_size)]
        return np.array(output_arr)