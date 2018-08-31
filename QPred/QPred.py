import cntk
import numpy as np
from DataConverter import convertData
import matplotlib.pyplot as plt

# Pandas datareader error workaround 
#import pandas as pd
#pd.core.common.is_list_like = pd.api.types.is_list_like
#from pandas_datareader import data as pddr

# TODO: Look into online data readers with pd 
#tickers = ['APPL']
#startDate = '2010-01-01'
#endDate = '2016-12-31'
#panelData = pddr.DataReader('INPX', 'google', startDate, endDate)

# Number of days ahead to look
timeShift   = 15

# Number of data points in a sequence
timeSteps   = 1000
# % Change we're looking at, up or down
threshold   = 0.04
dataPath    = './data/EOD-INTC.csv'

numEpochs   = 50
batchSize   = 16
numFeatures = 5
numClasses  = 3
lstmLayers  = 6

def createReader(filePath, isTraining, inputDim, outputDim):
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(filePath, cntk.io.StreamDefs(
        features = cntk.io.StreamDef(field='X', shape=inputDim,  is_sparse=False),
        labels   = cntk.io.StreamDef(field='Y', shape=outputDim, is_sparse=True),
        )), randomize=isTraining, max_sweeps=cntk.io.INFINITELY_REPEAT if isTraining else 1)
                                   

def createModel(input, numClasses, hiddenDim):
    model = cntk.layers.Sequential([
        cntk.layers.Recurrence(cntk.layers.LSTM(hiddenDim)),
        cntk.sequence.last,
        cntk.layers.Dense(numClasses)
        ])
    return model

# Going to start out trying to predict whether
# a stock will have changed (up or down) more than x% after n days
def train():

    # TODO: Need to add a method that reads exact sample size when
    # we're loading data that's already been converted
    #convertData(dataPath, 'intel', threshold, timeSteps, timeShift)


    input   = cntk.sequence.input_variable((numFeatures), name='features')
    label   = cntk.input_variable((numClasses), name='label')

    trainReader = createReader('./data/intel.ctf', True, numFeatures, numClasses)
    inputMap    = { 
        input: trainReader.streams.features, 
        label: trainReader.streams.labels 
    }

    model   = createModel(input, numClasses, lstmLayers)
    z       = model(input)

    loss    = cntk.cross_entropy_with_softmax(z, label)
    error   = cntk.classification_error(z, label)

    lrPerSample = cntk.learning_parameter_schedule_per_sample(0.03)

    learner     = cntk.adam(z.parameters, lrPerSample, 0.98)
    printer     = cntk.logging.ProgressPrinter(10, tag='Training')
    trainer     = cntk.Trainer(z, (loss, error), learner, [printer])

    samplesPerSeq   = 1000
    sequences       = 1992

    minibatchSize   = batchSize * samplesPerSeq
    minibatches     = sequences // batchSize

    for e in range(numEpochs):
        for b in range(minibatches):
            mb = trainReader.next_minibatch(minibatchSize, inputMap)
            trainer.train_minibatch(mb)
        trainer.summarize_training_progress()



    return

train()