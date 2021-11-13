import numpy as np
import scipy as sp
# import all
from Layers import *
from Optimization import *

class NeuralNetwork:

    def __init__(self, optimizer):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        pass

    def backward(self):
        pass

    def append_layer(self, layer):
        pass

    def train(self, iterations):
        pass

    def test(self, input_tensor):
        pass