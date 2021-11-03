import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        #check weather the checkerboard size is right
        if not ((self.resolution == self.tile_size * 8) or (self.resolution == self.tile_size * 10)):
            # todo: required action
            pass
        dim = int(self.resolution / self.tile_size)
        image_array = np.zeros((dim, dim), dtype=int)
        for m in range(dim):
            for n in range(dim):
                if (m + n) & 1 == 1:
                    image_array[m, n] = 1

        self.output = image_array

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    def __int__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        x, y = np.meshgrid(x, y)

        self.output = self.calculate_circle_area(self.position, self.radius, x, y)

    def calculate_circle_area(self, center, radius, x, y):
        calculated_area = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        for x in range(0, self.resolution):
            for y in range(0, self.resolution):
                if calculated_area[x, y] < radius:
                    calculated_area[x, y] = 1
                elif calculated_area[x, y] >= radius:
                    calculated_area[x, y] = 0
        return calculated_area

    def shaw(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Sprectrum:
    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        sprectrum_zeros = np.zeros([self.resolution, self.resolution, 3])

        sprectrum_zeros[:, :, 0] = np.linspace(0, 1, self.resolution)
        sprectrum_zeros[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        sprectrum_zeros[:, :, 2] = np.linspace(1, 0, self.resolution)

        self.output = sprectrum_zeros

    def show(self):
        plt.imshow(self.output)
        plt.show()

if __name__ == '__main__':
    # checker = Checker(250, 25)
    # checker.draw()
    # checker.show()
    sprec = Sprectrum(255)
    sprec.draw()
    sprec.show()