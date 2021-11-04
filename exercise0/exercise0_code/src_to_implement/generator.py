import os.path
import json
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # load json file for label data
        f = open(label_path)
        self.labels = list(json.load(f).items())

        # define epoch for each object
        self.current_epoch = 1
        self.used_data = set()
        self.tracker = 0

        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.label_len = len(self.labels)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        # update current epoch
        if len(self.used_data) >= self.label_len:
            self.current_epoch += 1 # new epoch
            self.used_data = set() # reset used data
            self.tracker = 0

        images = []
        labels_array = []
        batch = self.batch_size

        # then create a image list for a batch
        start = self.tracker
        end = self.tracker + self.batch_size
        end = end if end <= len(self.labels) else len(self.labels)
        one_time_check = True

        print(start, end)

        while start < end:
            label = self.labels[start]
            img_file = label[0] + '.npy'
            image_path = os.path.join(self.file_path, img_file)
            image = np.load(image_path)
            # TODO: augment image
            image = self.augment(image)
            # TODO: resize image based on image_size
            images.append(image)
            labels_array.append(label[1])
            self.used_data.add(label[0])
            start += 1

        if len(images) < batch:
            start, end = 0, batch - len(images)

        while start < end:
            label = self.labels[start]
            img_file = label[0] + '.npy'
            image_path = os.path.join(self.file_path, img_file)
            image = np.load(image_path)
            # TODO: augment image
            image = self.augment(image)
            # TODO: resize image based on image_size
            images.append(image)
            labels_array.append(label[1])
            self.used_data.add(label[0])
            start += 1

        self.tracker = end

        print(len(images), len(labels_array))

        return (images, labels_array)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        aug_img = img
        aug_mir_flag = True
        aug_rot_flag = False

        # for shuffling
        if self.shuffle:
            pass

        if self.mirroring:
            if not aug_mir_flag:
                aug_mir_flag = True
            else:
                aug_img = np.flipud(aug_img)
                aug_mir_flag = False

        if self.rotation:
            if not aug_mir_flag:
                aug_mir_flag = True
            else:
                deg = np.random.choice(np.linspace(90, 270, 3))
                aug_img = ndimage.rotate(aug_img, deg)
                aug_rot_flag = False

        return aug_img

    def current_epoch(self):
        # return the current epoch number
        return self.current_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        int_value = int(x)
        return self.class_dict[int_value]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        pass

if __name__ == '__main__':
    gen = ImageGenerator('exercise_data/', 'Labels.json', 12, [32, 32, 3])
    while True:
        gen.next()