from easydict import EasyDict
import json
import os
import yaml

class Config:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_config(self):
        return self.config