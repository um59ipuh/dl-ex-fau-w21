import numpy as np
from .Base import BaseLayer
import math
class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        BaseLayer.__init__(self)
        self.stride_shape_y, self.stride_shape_x = stride_shape[0], stride_shape[1]
        self.pooling_shape_y, self.pooling_shape_x = pooling_shape[0], pooling_shape[1]
        self.image_status = None
        self.input_tensor_shape = None
        self.output_index = []

    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        max_batch = []
        max_batch_index = []
        for img in range(0, batch_size):
            image = input_tensor[img]
            if len(image.shape) == 2:
                self.image_status = '1D'
            elif len(image.shape) == 3:
                self.image_status = '2D'
                channel = image.shape[0]
                row = image.shape[1]
                coloumn = image.shape[2]
                max_ch = []
                max_ch_index = []
                for ch in range(0, channel):
                    max_row = []
                    max_index = []
                    for i in range(0, math.floor(row / self.stride_shape_y)):
                        max_col = []
                        for j in range(0, math.floor(coloumn / self.stride_shape_x)):
                            start_x = j * self.stride_shape_x
                            end_x = start_x + self.pooling_shape_x
                            start_y = i * self.stride_shape_y
                            end_y = start_y + self.pooling_shape_y
                            if end_x <= coloumn and end_y <= row:
                                arr = image[ch][start_y:end_y, start_x:end_x]
                                maximum = np.amax(arr)
                                index = np.where(arr == maximum)
                                position_row = int(index[0][0]) + i * self.stride_shape_y
                                position_column = int(index[1][0]) + j * self.stride_shape_x
                                max_index.append([position_row, position_column])
                                max_col.append(maximum)

                            # maximum = np.amax(image[ch][start_y:end_y, start_x:end_x])
                            # max_col.append(maximum)
                        max_col_array = np.array(max_col)
                        #print("max_col_array: ",max_col_array.shape)
                        max_row.append(max_col_array)
                    max_row_array = np.array(max_row)
                    max_index_array = np.array(max_index)
                    max_ch_index.append(max_index_array)
                    #print("max_row_array: ", max_row_array.shape)
                    max_ch.append(max_row_array)
                max_ch_array = np.array(max_ch)
                max_ch_index_array = np.array(max_ch_index)
                max_batch_index.append(max_ch_index_array)
                max_batch.append(max_ch_array)
        self.output_index = np.array(max_batch_index)
        output = np.array(max_batch)
        # print("output index shape: ", self.output_index.shape)
        # print("output index: ", self.output_index)
        #print("output: ", output.shape)
        return output

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        new_array_batch = []
        for img in range(0, batch_size):
            error = error_tensor[img]
            channel = error.shape[0]
            new_array_ch = []
            for ch in range(0, channel):
                new_array = np.zeros((self.input_tensor_shape[-2], self.input_tensor_shape[-1]))
                row_shape = error.shape[-2]
                column_shape = error.shape[-1]
                i = 0
                for row in range(0, row_shape):
                    for col in range(0, column_shape):
                        y = self.output_index[img][ch][i][0]
                        x = self.output_index[img][ch][i][1]
                        i = i+1
                        new_array[y,x] = new_array[y,x] + error[ch][row][col]
                new_array_ch.append(new_array)
            new_array_ch_array = np.array(new_array_ch)
            new_array_batch.append(new_array_ch_array)

        error_prev = np.array(new_array_batch)


        return error_prev