import yaml


def get():
    return yaml.safe_load(open("qd/voronoielites/config.yml"))
