import tensorflow as tf
import os
import math
import random
import numpy as np


def read_image(path, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(
        image, [image_size, image_size])
    return image


class image_reader:
    files = []

    def __init__(self, root_path, image_size, generator):
        self.image_size = image_size
        self.update_all_file_paths(root_path)
        self.generator = generator
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
        index = 0
        random_label = random.random() * 0.5
        while True:
            rand = random.random()
            if rand >= 0.5:
                try:
                    path = self.files[index]
                    image = read_image(path, self.image_size)/255
                    x = tf.expand_dims(image, 0)
                    yield (x, np.array([0 + random_label]))
                except:
                    print(path)
                index += 1
            else:
                yield (self.generator.predict(), np.array([1 + random_label]))

            if index >= len(self.files):
                index = 0
