from tensorflow import keras
import numpy as np
from os import path


class generator:
    def __init__(self, latent_size, image_size, channels, weight_path):
        self.weight_path = weight_path
        self.latent_size = latent_size
        self.input = keras.Input(shape=(latent_size,))
        self.model = keras.layers.Dense(128 * 32 * 32)(self.input)

        self.model = keras.layers.LeakyReLU()(self.model)
        self.model = keras.layers.Reshape(
            (int(image_size/2), int(image_size/2), 128))(self.model)

        self.model = keras.layers.Conv2D(256, 5, padding='same')(self.model)
        self.model = keras.layers.LeakyReLU()(self.model)

        self.model = keras.layers.Conv2DTranspose(
            256, 4, strides=2, padding='same')(self.model)
        self.model = keras.layers.LeakyReLU()(self.model)

        self.model = keras.layers.Conv2D(256, 5, padding='same')(self.model)
        self.model = keras.layers.LeakyReLU()(self.model)
        self.model = keras.layers.Conv2D(256, 5, padding='same')(self.model)
        self.model = keras.layers.LeakyReLU()(self.model)

        self.model = keras.layers.Conv2D(
            channels, 7, activation='tanh', padding='same')(self.model)
        self.model = keras.models.Model(self.input, self.model)
        self.load_weight()

    def summary(self):
        self.model.summary()

    def predict(self):
        random_latent_vectors = np.random.normal(
            size=(1, self.latent_size))
        return self.model.predict(random_latent_vectors)

    def save_weight(self):
        """ save weight """
        self.model.save_weights(self.weight_path)
        print('weights were saved.')

    def load_weight(self):
        """ load weights """
        if path.exists(self.weight_path):
            print("already have weight {}".format(self.weight_path))
            self.model.load_weights(self.weight_path)
        else:
            print("not found {}".format(self.weight_path))
        return self
