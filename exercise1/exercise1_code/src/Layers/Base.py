import numpy as np

# Base class for training layers
class BaseLayer(object):
    def __init__(self, weights = []):
        self.trainable = False
        self.weights = weights
