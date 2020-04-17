from src.configure import configure
from src.generator import generator
from src.discriminator import discriminator
from src.gan import gan
import matplotlib.pyplot as plt
from src.image_reader import image_reader

""" set up config """
config = configure()
data_folder_path = config.source['data_folder_path']
gen_weight_file_path = config.source['gen_weight_file_path']
dis_weight_file_path = config.source['dis_weight_file_path']
image_size = int(config.setting['image_size'])
epochs = int(config.setting['epochs'])
steps_per_epoch = int(config.setting['steps_per_epoch'])
latent_size = int(config.setting['latent_size'])
channels = int(config.setting['channels'])
iterations = int(config.setting['iterations'])

""" set up generator """
generator_object = generator(
    latent_size, image_size, channels, gen_weight_file_path)
generator_model = generator_object.model
generator_object.summary()

""" predict """
image = generator_object.predict()[0]

print(image)

""" display image """
plt.title("Result")
plt.imshow(image)
plt.show()
