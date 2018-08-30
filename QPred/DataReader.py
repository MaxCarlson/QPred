import numpy as np

class DataReader():
    def __init__(self, delimiter=',', eol='\n'):
        self.delimiter = delimiter
        self.eol = eol
        
