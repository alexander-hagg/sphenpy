import yaml


def get():
    return yaml.safe_load(open("qd/mapelites/config.yml"))
