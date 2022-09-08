import yaml


def get():
    return yaml.safe_load(open("sphen/config.yml"))
