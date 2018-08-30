import numpy as np

def fromFile(filePath):
    data = np.loadtxt(filePath, dtype=np.str, delimiter=',', skiprows=1)
    return data

# TODO: Actually we don't want to normalize all the data
# to one, we want to normalize it inside sequences! I think!
def normalize(data):
    for f in range(np.size(data, 1)):
        l = data[:,f]
        min = np.min(data[:,f])
        max = np.max(data[:,f])



# This is specific for our current datasets
def convertData(filePath, timeSteps, timeShift):
    data    = fromFile(filePath)

    # TODO: Possibly fill in missing dates (weekends/holdiays)
    # with data based on prior prices?
    dates   = data[:, 0:1]
    data    = data[:, 1:].astype(np.float)

    normalize(data)

    a = 5

