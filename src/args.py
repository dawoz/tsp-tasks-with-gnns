import dataclasses

import yaml


def load_config(file_path):
    """Load a YAML file"""
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def override_args(default, args):
    """Override the default arguments with the provided arguments"""
    default = dataclasses.replace(default)
    for key, value in args.items():
        if value is not None:
            setattr(default, key, value)
    return default
