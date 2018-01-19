from configparser import ConfigParser

class Config(ConfigParser):
    """docstring for Config"""
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __call__(self, filenames, encoding=None):
        self.read(filenames, encoding=None)
        return self

