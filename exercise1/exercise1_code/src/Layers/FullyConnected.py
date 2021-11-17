import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        BaseLayer.__init__(self)
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(self.input_size, self.output_size) # generates value [0, 1)

        # set the default optimizer as SGD
        self._optimizer = 'sgd'

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        return np.dot(input_tensor, self.weights)

    def backward(self, error_tensor):
        return np.dot(self.optimizer.transpose(), error_tensor)
#serves as the input tensor for the next layer. input tensor is a matrix with input size columns and batch size rows. 
#The batch size represents the number of inputs processed simultaneously. The output size is a parameter of the layer specifying the number of
#columns of the output
# returns a tensor that serves as the error tensor for the previous layer. Quick reminder: in the backward pass we are
#going in the other direction as in the forward pass.
#Use the method calculate update(weight tensor, gradient tensor) of your optimizer in your backward pass, in order to update your weights. 
#Don’t perform an update if the optimizer is not set.
