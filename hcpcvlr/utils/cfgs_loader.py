import yaml


def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfgs = yaml.load(f.read(), Loader=yaml.FullLoader)
        return cgfs

def load_json(json_path):
    pass
