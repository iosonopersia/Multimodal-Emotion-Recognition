from munch import Munch

config = None

def get_config():
    global config
    if config is None:
        with open('./src/config.yaml', 'rt', encoding='utf-8') as f:
            config = Munch.fromYAML(f.read())
    return config
