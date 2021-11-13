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

    def backword(self, error_tensor):
        pass