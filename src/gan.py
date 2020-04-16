import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np


class gan:
    def __init__(self, discriminator_model, generator_model, latent_size, iterations=1000):
        self.latent_size = latent_size
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

    def train_generator(self, epochs, steps_per_epoch):
        batch_size = epochs * steps_per_epoch
        random_latent_vectors = np.random.normal(size=(batch_size,
                                                       self.latent_size))
        misleading_targets = np.zeros(batch_size)
        for latent, y in zip(random_latent_vectors, misleading_targets):
            extend_latent = tf.expand_dims(latent, 0)
            yield (np.array(extend_latent), np.array([y]))

    def train(self, epochs, steps_per_epoch):
        a_loss = self.gan.fit(self.train_generator(epochs, steps_per_epoch), epochs=epochs,
                              steps_per_epoch=steps_per_epoch)
        return self
