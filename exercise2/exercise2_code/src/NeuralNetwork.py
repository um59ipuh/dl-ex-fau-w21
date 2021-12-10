import copy


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.input_tensor = self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            output_tensor = layer.forward(self.input_tensor)
            # input for next layer
            self.input_tensor = output_tensor

        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        return loss

    def backward(self):
        error_value = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            output_tensor = layer.backward(error_value)
            error_value = output_tensor

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        output_tensor = None
        for layer in self.layers:
            output_tensor = layer.forward(input_tensor)
            input_tensor = output_tensor
        return output_tensor