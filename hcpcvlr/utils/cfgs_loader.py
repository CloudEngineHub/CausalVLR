import yaml


def load_yaml(yaml_path, cfgs=None):
    """
    loading config file .yaml
    and extend the dict.
    """
    if cfgs is None:
        cfgs = {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        _cfgs = yaml.load(f.read(), Loader=yaml.FullLoader)
        for key in _cfgs.keys():
            if key == '__base__':
                cfgs = load_yaml(_cfgs[key], cfgs)
            else:
                cfgs.update(_cfgs[key])
    return cfgs

def load_json(json_path):
    pass
