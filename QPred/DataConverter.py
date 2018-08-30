import numpy as np

closeIdx = 3

def fromFile(filePath):
    data = np.loadtxt(filePath, dtype=np.str, delimiter=',', skiprows=1)
    return data

# TODO: Actually we don't want to normalize all the data
# to one, we want to normalize it inside sequences! I think!
def normalize(data):
    def maxMin(min, minMax, x):
        return  (x - min) / minMax

    for f in range(np.size(data, 1)):
        min = np.min(data[:,f])
        max = np.max(data[:,f])
        minMax = max - min
        
        data[:,f] = np.array([maxMin(min, minMax, x) for x in data[:,f]])


def calcLabel(threshold, lastX, Y):
    change = (Y[closeIdx] - lastX[closeIdx]) / lastX[closeIdx]
    if change >= threshold:
        return 2
    elif change <= -threshold:
        return 0
    return 1

# TODO: Another question, should our sequences overlap at all?
def toSequences(data, threshold, timeSteps, timeShift, seqDist):
    numSeq = (len(data) - timeSteps + timeShift) // seqDist

    seqs = []

    for s in range(0, numSeq*seqDist, seqDist):
        X = data[s:s + seqDist]
        Y = calcLabel(threshold, data[s+seqDist-1], data[s+seqDist+timeShift])
        seqs.append([X, Y])

    return seqs

# This is specific for our current datasets
def convertData(filePath, threshold, timeSteps, timeShift):
    data    = fromFile(filePath)

    # TODO: Possibly fill in missing dates (weekends/holdiays)
    # with data based on prior prices?
    dates   = data[:, 0:1]
    data    = data[:, 1:].astype(np.float)

    data    = toSequences(data, threshold, timeSteps, timeShift, 5)

    normalize(data)

    a = 5

