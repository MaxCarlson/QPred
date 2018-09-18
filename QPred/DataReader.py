import numpy as np

class DataReader():
    def __init__(self, path, numFeatures, numClasses, batchSize, seqLen, random):
        self.path       = path
        self.delimiter  = ' '
        self.batchSize  = batchSize
        self.seqLen     = seqLen
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
        while True:
            seqIdx      = 0
            batchIdx    = 0
            feats       = np.zeros((self.batchSize, self.seqLen, self.numFeatures), dtype=np.float32)
            labs        = np.zeros((self.batchSize, self.numClasses), dtype=np.float32)

            for l in self.file:
                blocks = l.split()

                # New sequence
                if seqIdx == self.seqLen:
                   batchIdx += 1
                   seqIdx    = 0

                # If our batch has been filled yield the batch
                if batchIdx == self.batchSize:
                    yield feats, labs
                    labs.fill(0)
                    feats.fill(0)
                    
                    seqIdx      = 0
                    batchIdx    = 0
                

                # Add features to array
                featIdx = 0
                for i in blocks[2:8]:
                    feats[batchIdx, seqIdx, featIdx] = np.float(i)
                    featIdx += 1

                # If we're looking at the first set of features 
                # in a sequence than the last element is the sparse label

                if seqIdx == 0:
                    labs[batchIdx, int(blocks[-1][0])] = 1.

                seqIdx += 1

            # Reset file to line 0
            self.file.seek(0)




                

