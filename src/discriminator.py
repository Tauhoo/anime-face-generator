import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class discriminator:
    def __init__(self, height, width, channels):
        self.input = layers.Input(shape=(height, width, channels))
        self.model = layers.Conv2D(128, 3)(self.input)
        self.model = layers.LeakyReLU(0.2)(self.model)
        self.model = layers.Conv2D(128, 4, strides=2)(self.model)
        self.model = layers.LeakyReLU(0.2)(self.model)
        self.model = layers.Conv2D(128, 4, strides=2)(self.model)
        self.model = layers.LeakyReLU()(self.model)
        self.model = layers.Conv2D(128, 4, strides=2)(self.model)
        self.model = layers.LeakyReLU()(self.model)

        self.model = layers.Flatten()(self.model)

        self.model = layers.Dropout(0.4)(self.model)

        self.model = layers.Dense(1, activation='sigmoid')(self.model)

        self.model = keras.models.Model(self.input, self.model)

    def summary(self):
        self.model.summary()
