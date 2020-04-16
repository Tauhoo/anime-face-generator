import configparser


class configure:
    def __init__(self):
        config = configparser.ConfigParser(allow_no_value=True)
        config_text = open('./config.ini').read()
        config.read_string(config_text)
        self.setting = config["setting"]
        self.source = config["source"]
