import numpy as np

class DataReader():
    def __init__(self, path, features, labels, numFeatures, numClasses, batchSize, random):
        self.path       = path
        self.delimiter  = ' '
        self.random     = random
        self.features   = features
        self.labels     = labels
        self.numFeatures = numFeatures
        self.numClasses = numClasses
        self.file       = open(path, "r")
        self.outFunc    = self.randomGen() if random else self.normalGen()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.outFunc)

    def randomGen(self):
        pass

    def normalGen(self):

        fbatch = np.zeros(1)

        while True:
            seqId = 0

            for l in self.file:
                blocks = l.split(self.delimiter)

                if seqId != blocks[0]:
                    pass
                else:

