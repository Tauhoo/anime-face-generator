from src.configure import configure
from src.generator import generator
from src.discriminator import discriminator
from src.gan import gan
from src.image_reader import image_reader

""" set up config """
config = configure()
data_folder_path = config.source['data_folder_path']
weight_file_path = config.source['weight_file_path']
image_size = int(config.setting['image_size'])
epochs = int(config.setting['epochs'])
steps_per_epoch = int(config.setting['steps_per_epoch'])
latent_size = int(config.setting['latent_size'])
channels = int(config.setting['channels'])
iterations = int(config.setting['iterations'])

""" set up generator """
generator_object = generator(latent_size, image_size, channels)
generator_model = generator_object.model
generator_object.summary()

""" image reader """
image_reader = image_reader(data_folder_path, image_size, generator_object)
train_data = image_reader.get_train_generator()

""" set up discriminator """
discriminator_object = discriminator(image_size, image_size, channels)
discriminator_model = discriminator_object.model
discriminator_object.summary()

""" set up gan """
gan_object = gan(discriminator_model, generator_model, latent_size, iterations)

""" train """
