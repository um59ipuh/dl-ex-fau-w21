import numpy as np
import Base

class FullyConnected(Base):

    def __init__(self, input_size, output_size):
        Base.__init__(self)
        self.trainable = True
        self.weights = np.random.rand() # generates value [0, 1)

        # set the default optimizer as SGD
        self._optimizer = 'sgd'

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        pass
#serves as the input tensor for the next layer. input tensor is a matrix with input size columns and batch size rows. 
#The batch size represents the number of inputs processed simultaneously. The output size is a parameter of the layer specifying the number of
#columns of the output


    def backword(self, error_tensor):
        pass
# returns a tensor that serves as the error tensor for the previous layer. Quick reminder: in the backward pass we are
#going in the other direction as in the forward pass.
#Use the method calculate update(weight tensor, gradient tensor) of your optimizer in your backward pass, in order to update your weights. 
#Don’t perform an update if the optimizer is not set.
