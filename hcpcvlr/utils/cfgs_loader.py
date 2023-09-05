import yaml


def load_yaml(yaml_path):
    """
    loading config file .yaml
    and extend the dict.
    """
    cfgs = {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        _cfgs = yaml.load(f.read(), Loader=yaml.FullLoader)
        for key in _cfgs.keys():
            cfgs.update(_cfgs[key])
    return cfgs

def load_json(json_path):
    pass
