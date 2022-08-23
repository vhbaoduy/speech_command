import yaml


class Configuration():
    def __init__(self, yaml_path):
        yaml_config_file = open(yaml_path)
        self._attribute = yaml.load(yaml_config_file, Loader=yaml.FullLoader)['settings']

    def __str__(self):
        print("#" * 5, "DATASET CONFIGURATION INFO", "#" * 5)
        pretty(self._attribute)
        print("#" * 50)
        return '\n'

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, item):
        try:
            return self._attribute[item]
        except KeyError:
            try:
                return self.__dict__[item]
            except KeyError:
                return None


def pretty(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key) + ":", end='')
        if isinstance(value, dict):
            print()
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


if __name__ == '__main__':
    config = Configuration("configs.yaml")
    print(config)


