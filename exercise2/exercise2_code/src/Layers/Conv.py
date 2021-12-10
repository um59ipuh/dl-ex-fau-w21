import numpy as np
import scipy.signal
from scipy import signal
from .Base import BaseLayer
import math
from scipy.ndimage import correlate1d
import copy


class Conv(BaseLayer):
    def __init__(self,  stride_shape, convolution_shape, num_kernels):
        BaseLayer.__init__(self)
        self.trainable = True

        if len(stride_shape) == 1:
            self.stride_shape_y, self.stride_shape_x = stride_shape[0], stride_shape[0]
            self.num_kernels = num_kernels
        elif len(stride_shape) == 2:
            self.stride_shape_y, self.stride_shape_x = stride_shape[0], stride_shape[1]

        self.num_kernels = num_kernels
        self.convolution_shape = convolution_shape
        if len(convolution_shape) == 2:
            self.conv_c = convolution_shape[0]
            self.conv_m = convolution_shape[1]
            self.conv_n = 1
            self._weights = np.empty([num_kernels, self.conv_c, self.conv_m])
            for i in range (0, self.num_kernels):
                self._weights[i] = np.random.rand(self.conv_c, self.conv_m)
        elif len(convolution_shape) == 3:
            self.conv_c = convolution_shape[0]
            self.conv_m = convolution_shape[1]
            self.conv_n = convolution_shape[2]
            self._weights = np.empty([num_kernels, self.conv_c, self.conv_m, self.conv_n])
            for i in range (0, self.num_kernels):
                self._weights[i] = np.random.rand(self.conv_c, self.conv_m, self.conv_n)
        bias = []
        for i in range(0, self.num_kernels):
            bias.append(float(np.random.rand(1)))
        self._bias = np.array(bias)
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.image_status = None
        self.input_tensor_shape = None
        self.past_grad_weights = 0
        self.past_grad_bias = 0
        self.bias_optimizer = None
        self.weights_optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    @property
    def bias(self):
        #self.bias_optimizer = copy.deepcopy(self.optimizer)
        return self._bias

    @bias.setter
    def bias(self, bias):
        #self.bias_optimizer = copy.deepcopy(self.optimizer)
        self._bias = bias

    @property
    def weights(self):
        #self.weights_optimizer = copy.deepcopy(self.optimizer)
        return self._weights

    @weights.setter
    def weights(self, weights):
        #self.weights_optimizer = copy.deepcopy(self.optimizer)
        self._weights = weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.weights_optimizer = copy.deepcopy(self._optimizer)
        self.bias_optimizer = copy.deepcopy(self._optimizer)


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor_shape = input_tensor.shape
        if len(input_tensor.shape) == 3:
            self.image_status = "1D"
        elif len(input_tensor.shape) == 4:
            self.image_status = "2D"
        batch_size = input_tensor.shape[0] #(b,c,y) or (b,c,y,x)
        next_input_tensor = []
        for img in range(0, batch_size):
            input = input_tensor[img] #(c,y) or (c,y,x)
            all_kernel = []
            for kernel in range(0, self.num_kernels):
                n = signal.correlate(input, self._weights[kernel], mode='same')
                if self.image_status == "1D":
                    if self.conv_c <= 2:
                        next = np.add(n[self.conv_c-1], self._bias[kernel]).reshape(1,-1)
                    else:
                        n_ch = math.floor(self.conv_c / 2)
                        next = np.add(n[n_ch], self._bias[kernel]).reshape(1,-1)
                if self.image_status == "2D":
                    if self.conv_c <= 2:
                        next = np.add(n[self.conv_c-1], self._bias[kernel])
                    else:
                        n_ch = math.floor(self.conv_c/2)
                        next = np.add(n[n_ch], self._bias[kernel])
                # Down Sampling
                sampled_array = next[0:next.shape[-2]:self.stride_shape_y, 0:next.shape[-1]:self.stride_shape_x]
                if self.image_status == "1D":
                    sampled_array = sampled_array.squeeze(axis=0)
                else:
                    pass
                all_kernel.append(sampled_array)   #(k,y,x)
            all_kernel_array = np.array(all_kernel)
            next_input_tensor.append(all_kernel_array) #(b,k,y,x)
        next_input_tensor_array = np.array(next_input_tensor)

        return next_input_tensor_array

    def backward(self, error_tensor):

        batch_size = error_tensor.shape[0]  # (b,k,y,x)
        error_batch = []
        ## Weight_common
        weights_all = []
        for channel in range(0, self.conv_c):
            w = []
            for kernel in range(0, self.num_kernels):    #for ch1 : k1ch1, k2ch1... for ch2: k1ch2, k2ch2...
                w.append(self._weights[kernel, channel])
            weight_per_channel = np.array(w)
            weights_all.append(weight_per_channel)
        weights_backward = np.array(weights_all)

        ## upsampling
        upsampled_error_batch = []
        for img in range(0, batch_size):
            error = error_tensor[img]  # (k,y,x)
            if self.image_status == "2D":
                channel = error.shape[0]
                new_array_ch = []
                for ch in range(0, channel):
                    new_array = np.zeros((self.input_tensor_shape[-2], self.input_tensor_shape[-1]))
                    row_shape = error.shape[-2]
                    column_shape = error.shape[-1]
                    for row in range(0, row_shape):
                        y = 0
                        x = 0
                        for col in range(0, column_shape):
                            y = row * self.stride_shape_y
                            x = col * self.stride_shape_x
                            new_array[y,x] = error[ch][row][col]
                    new_array_ch.append(new_array)
                upsampled_error_ch = np.array(new_array_ch)
                upsampled_error_batch.append(upsampled_error_ch)
            elif self.image_status == "1D":
                channel = error.shape[0]
                new_array_ch = []
                for ch in range(0, channel):
                    new_array = np.zeros((self.input_tensor_shape[-1]))
                    column_shape = error.shape[-1]
                    x = 0
                    for col in range(0, column_shape):
                        x = col * self.stride_shape_x
                        new_array[x] = error[ch][col]
                    new_array_ch.append(new_array)
                upsampled_error_ch = np.array(new_array_ch)
                upsampled_error_batch.append(upsampled_error_ch)
        upsampled_error = np.array(upsampled_error_batch)

        ## Calculation of En-1
        for img in range(0, batch_size):
            e1 = []
            for channel in range(0, self.conv_c):
                e = scipy.signal.convolve(upsampled_error[img], np.flip(weights_backward[channel],0), mode='same') #e(k,y,x) * weight(k,y,x) = (k,y,x)
                if self.num_kernels <= 2:
                    e1.append(e[self.num_kernels-1])
                else:
                    e_ch = math.floor(self.num_kernels / 2)
                    e1.append(e[e_ch])
            error_ch = np.array(e1)
            error_batch.append(error_ch)
        error_prev = np.array(error_batch)

        ## calculation of gradient weights
        added_gradient = np.zeros((self.weights.shape))
        added_bias = np.zeros((self.bias.shape))
        for img in range(0, batch_size):
            error = error_tensor[img] #(c,y,x)
            # Gradient calculation
            combined_kernel = []
            bias = []
            for kernel in range(0, self.num_kernels):
                combined_channel = []
                for channel in range(0, self.conv_c):
                    if self.image_status == "1D":
                        in_tensor = self.input_tensor[img, channel]
                        if self.stride_shape_y == 1:
                            out = correlate1d(in_tensor, weights=error[kernel])
                        else:
                            out = signal.correlate2d(in_tensor.reshape(-1,1), upsampled_error[img][kernel].reshape(-1,1), mode='valid')[:,0]
                    elif self.image_status == "2D":
                        pad_m = math.floor(self.conv_m / 2)
                        pad_n = math.floor(self.conv_n / 2)
                        padded_input = np.pad(self.input_tensor[img, channel], ((pad_m, pad_m), (pad_n, pad_n)),
                                              mode='constant', constant_values=(0, 0))
                        out_pre = signal.correlate2d(padded_input, upsampled_error[img][kernel], mode='valid')  # input_tensor(y,x) conv error(y,x)
                        out = out_pre[:self.weights.shape[2], :self.weights.shape[3]]
                    combined_channel.append(out)
                combined_channel_array = np.array(combined_channel)
                combined_kernel.append(combined_channel_array)
                #for bias
                bias.append(np.sum(error[kernel]))
            bias_array = np.array(bias)
            added_bias = np.add(added_bias, bias_array)
            combined_kernel_array_img = np.array(combined_kernel)
            added_gradient = np.add(added_gradient, combined_kernel_array_img)
        self.gradient_weights = added_gradient
        self.gradient_bias = added_bias

        if self._optimizer != None:
            self._weights = self.weights_optimizer.calculate_update(self._weights, self.gradient_weights)
            self._bias = self.bias_optimizer.calculate_update(self._bias, self.gradient_bias)
        return error_prev

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = self._weights.shape
        bias_shape = self._bias.shape
        fan_in = self.conv_c * self.conv_m * self.conv_n
        fan_out = self.num_kernels * self.conv_m * self.conv_n
        self._weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        self._bias = bias_initializer.initialize(bias_shape, fan_in, fan_out)


