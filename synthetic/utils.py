import yaml

def load_yaml(filename):
    """
    Load and print YAML config files
    """
    with open(filename, 'r') as stream:
        file = yaml.safe_load(stream)
        return file