import copy
import numpy as np

closeIdx    = 3 # Index of closing price, this is used to calculate our label
seqDist     = 2 # Distance each sequence is separated from the one before
writeTo     = './data/'

def fromFile(filePath):
    data = np.loadtxt(filePath, dtype=np.str, delimiter=',', skiprows=1)
    return data

# Normalize features in each sequence
# to other data points in the same feature/sequence
def normalize(data):
    def maxMin(min, minMax, x):
        return (x - min) / minMax

    for s in data:
        ss = s[0]
        for f in range(np.size(ss, 1)):
            feats   = ss[:,f]
            min     = np.min(feats)
            max     = np.max(feats)
            minMax  = max - min
        
            ss[:,f]   = np.array([maxMin(min, minMax, x) for x in feats])

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
        if s + timeSteps + timeShift >= len(data):
            break
        end = s + timeSteps
        X   = copy.copy(data[s:end])
        Y   = calcLabel(threshold, data[end-1], data[end+timeShift])
        seqs.append([X, Y])

    return seqs

def writeCtf(destName, data):
    aa = len(data)

    file = open(writeTo + destName + '.ctf', "w+")

    sId = 0
    for seq in data:
        i = 0
        for t in seq[0]:
            fStr = str(sId) +  ' |X ' + str(t)[1:-1].replace(',', ' ').replace('\n','')
            if i == 0:
                fStr += ' |Y ' + str(seq[1]) + ':1'
            i += 1
            file.write(fStr + '\n')

        sId += 1



# This is specific for our current datasets
# TODO: Add in test to data split
def convertData(filePath, destName, threshold, timeSteps, timeShift, split=[0.9,0.1]):
    data    = fromFile(filePath)

    data = data[0:5000]

    # TODO: Possibly fill in missing dates (weekends/holdiays)
    # and include dates, or relative dates into features?
    # with data based on prior prices?
    dates   = data[:, 0:1]
    data    = data[:, 1:].astype(np.float)

    # Remove adjusted data, dividend, and splits
    data    = data[:,0:5]
    data    = toSequences(data, threshold, timeSteps, timeShift, seqDist)

    normalize(data)

    
    # TODO: This presents an intersting question on how to split data when
    # sequences overlap. Should it be purely based on time ensuring no overlap?
    # For now we'll just split it based on dates, but in the future we should look back into this
    # Also, there's some overlap if we just do a simple split of the data here
    trainLen = int(len(data) * split[0])

    writeCtf(destName + '_train', data[0:trainLen])
    writeCtf(destName + '_valid', data[trainLen:])

