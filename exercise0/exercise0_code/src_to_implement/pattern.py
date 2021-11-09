import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        dim_show = int(self.resolution / self.tile_size)
        dim = self.resolution // (2 * self.tile_size)

        tile_two = 2 * self.tile_size
        tile_arr = np.zeros(tile_two, dtype=int)
        tile_arr[self.tile_size : 2*self.tile_size] = 1

        pre_output = np.tile(tile_arr, dim)
        output_arr = np.tile(pre_output, (self.resolution, 1))

        viz_array = np.zeros((dim_show, dim_show), dtype = int)
        viz_array[::2, 1::2] = 1
        viz_array[1::2, ::2] = 1
        self.vizualize = viz_array

        self.output = output_arr

        return self.output.copy()

    def show(self):
        plt.imshow(self.vizualize, cmap='gray')
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        x, y = np.meshgrid(x, y)

        self.output = self.calculate_circle_area(self.position, self.radius, x, y)

        return self.output.copy()

    def calculate_circle_area(self, center, radius, x, y):
        calculated_area = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        calculated_area[calculated_area <= self.radius] = 1
        calculated_area[calculated_area > self.radius] = 0
        return calculated_area

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        sprectrum_zeros = np.zeros([self.resolution, self.resolution, 3])

        sprectrum_zeros[:, :, 0] = np.linspace(0, 1, self.resolution)
        sprectrum_zeros[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        sprectrum_zeros[:, :, 2] = np.linspace(1, 0, self.resolution)

        self.output = sprectrum_zeros

        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()