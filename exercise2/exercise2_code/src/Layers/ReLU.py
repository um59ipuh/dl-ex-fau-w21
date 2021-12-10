import numpy as np
from .Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.class_input_tensor = []

    def forward(self, input_tensor):
        self.class_input_tensor=input_tensor
        zero_tensor = np.zeros(shape=np.shape(input_tensor))
        output_tensor = np.maximum(input_tensor, zero_tensor)

        return output_tensor

    def backward(self, error_tensor):

        updated_error_tensor = np.where(self.class_input_tensor<=0, 0, error_tensor)

        return updated_error_tensor

