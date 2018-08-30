import numpy as np

closeIdx = 3
writeTo = './data/'

def fromFile(filePath):
    data = np.loadtxt(filePath, dtype=np.str, delimiter=',', skiprows=1)
    return data

# Normalize features in each sequence
# to other data points in the same feature/sequence
def normalize(data):
    def maxMin(min, minMax, x):
        return  (x - min) / minMax

    for s in data:
        ss = s[0]
        for f in range(np.size(s[0], 1)):
            min = np.min(s[0][:,f])
            max = np.max(s[0][:,f])
            minMax = max - min
        
            s[0][:,f] = np.array([maxMin(min, minMax, x) for x in s[0][:,f]])


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
        X   = data[s:end]
        Y   = calcLabel(threshold, data[end-1], data[end+timeShift])
        seqs.append([X, Y])

    return seqs

def writeCtf(destName, data):
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
def convertData(filePath, destName, threshold, timeSteps, timeShift):
    data    = fromFile(filePath)

    #data = data[0:1200]


    # TODO: Possibly fill in missing dates (weekends/holdiays)
    # with data based on prior prices?
    dates   = data[:, 0:1]
    data    = data[:, 1:].astype(np.float)


    # Remove adjusted data, dividend, and splits
    data    = data[:,0:5]
    data    = toSequences(data, threshold, timeSteps, timeShift, 5)

    normalize(data)
    writeCtf(destName, data)

    a = 5

