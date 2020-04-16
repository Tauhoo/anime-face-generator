import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam, RMSprop


class gan:
    def __init__(self, discriminator_model, generator_model, latent_size, iterations=1000):
        self.iterations = iterations
        self.discriminator = discriminator_model
        self.generator = generator_model

        self.discriminator.trainable = False

        gan_input = keras.Input(shape=(latent_size,))
        gan_output = self.discriminator(self.generator(gan_input))
        self.gan = keras.models.Model(gan_input, gan_output)

        gan_optimizer = keras.optimizers.RMSprop(
            lr=0.0004, clipvalue=1.0, decay=1e-8)
        self.gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

        discriminator_optimizer = keras.optimizers.RMSprop(
            lr=0.0008,
            clipvalue=1.0,
            decay=1e-8
        )
        self.discriminator.compile(
            optimizer=discriminator_optimizer, loss='binary_crossentropy')

    def train(self):
        pass
