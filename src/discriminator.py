import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from os import path


class discriminator:
    def __init__(self, height, width, channels, weight_path):
        self.weight_path = weight_path
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
        self.load_weight()

    def summary(self):
        self.model.summary()

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

    def train(self, train_data, epochs, steps_per_epoch):
        self.model.trainable = True
        self.model.fit(train_data, epochs=epochs,
                       steps_per_epoch=steps_per_epoch)
