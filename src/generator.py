import tensorflow.keras as keras
import tensorflow.nn.layers as layers


class generator:
    def __init__(self, latent_size, channels):
        self.input = keras.Input(shape=(latent_size,))
        self.model = layers.Dense(128 * 32 * 32)(self.input)

        self.model = layers.LeakyReLU()(self.model)
        self.model = layers.Reshape((32, 32, 128))(self.model)

        self.model = layers.Conv2D(256, 5, padding='same')(self.model)
        self.model = layers.LeakyReLU()(self.model)

        self.model = layers.Conv2DTranspose(
            256, 4, strides=2, padding='same')(self.model)
        self.model = layers.LeakyReLU()(self.model)

        self.model = layers.Conv2D(256, 5, padding='same')(self.model)
        self.model = layers.LeakyReLU()(self.model)
        self.model = layers.Conv2D(256, 5, padding='same')(self.model)
        self.model = layers.LeakyReLU()(self.model)

        self.model = layers.Conv2D(
            channels, 7, activation='tanh', padding='same')(self.model)
        self.model = keras.models.Model(self.input, self.model)

    def summary(self):
        self.model.summary()
