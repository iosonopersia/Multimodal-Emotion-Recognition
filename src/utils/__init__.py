from munch import Munch
import yaml


config = None

def get_config():
    global config
    if config is None:
        with open('./src/config.yaml', 'rt', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = Munch.fromDict(config)
    return config
