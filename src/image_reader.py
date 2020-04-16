import tensorflow as tf
import os
import math
import random


def read_image(path, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(
        image, [image_size, image_size])
    return image


class image_reader:
    files = []

    def __init__(self, root_path, image_size):
        self.image_size = image_size
        self.update_all_file_paths(root_path)
        random.shuffle(self.files)

    def update_all_file_paths(self, root_path):
        paths = os.listdir(root_path)
        for path in paths:
            concat_path = root_path + '/' + path
            if os.path.isdir(concat_path):
                self.update_all_file_paths(concat_path)
            else:
                self.files.append(concat_path)

    def get_train_generator(self):
        while True:
            for path in self.files:
                try:
                    image = read_image(path, self.image_size)/255
                    x = tf.expand_dims(image, 0)
                    yield (x, image)
                except:
                    print(path)
