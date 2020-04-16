from src.configure import configure

config = configure()
data_folder_path = config.source['data_folder_path']
weight_file_path = config.source['weight_file_path']
image_size = config.setting['image_size']
epochs = config.setting['epochs']
steps_per_epoch = config.setting['steps_per_epoch']

print(data_folder_path)
