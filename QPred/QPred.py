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


dataPath    = './data/EOD-INTC.csv'

# Data generation parameters
timeShift   = 15    # Number of days ahead to look
timeSteps   = 1000  # Number of data points in a sequence
threshold   = 0.04  # % Change we're looking at, up or down
numFeatures = 5
numClasses  = 3

numEpochs   = 50
batchSize   = 16

lstmLayers  = 2
lstmSize    = 64

def createReader(filePath, isTraining, inputDim, outputDim):
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(filePath, cntk.io.StreamDefs(
        features = cntk.io.StreamDef(field='X', shape=inputDim,  is_sparse=False),
        labels   = cntk.io.StreamDef(field='Y', shape=outputDim, is_sparse=True),
        )), randomize=isTraining, max_sweeps=cntk.io.INFINITELY_REPEAT if isTraining else 1)
                                   

def createModel(input, numClasses, layers, lstmLayers):
    model = cntk.layers.Sequential([
        cntk.layers.For(range(layers), lambda: 
                   cntk.layers.Sequential([
                       cntk.layers.Stabilizer(), 
                       cntk.layers.Recurrence(cntk.layers.LSTM(lstmLayers), go_backwards=False)
                   ])),
        cntk.sequence.last,
        cntk.layers.Dropout(0.15),
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

    trainReader = createReader('./data/intel_train.ctf', True, numFeatures, numClasses)
    validReader = createReader('./data/intel_valid.ctf', True, numFeatures, numClasses)

    trainInputMap    = { 
        input: trainReader.streams.features, 
        label: trainReader.streams.labels 
    }

    validInputMap   = {
        input: validReader.streams.features,
        label: validReader.streams.labels
    }

    model   = createModel(input, numClasses, lstmLayers, lstmSize)
    z       = model(input)

    loss    = cntk.cross_entropy_with_softmax(z, label)
    error   = cntk.classification_error(z, label)

    lrPerSample = cntk.learning_parameter_schedule_per_sample(0.1)

    learner     = cntk.adam(z.parameters, lrPerSample, 0.98)
    printer     = cntk.logging.ProgressPrinter(20, tag='Training')
    trainer     = cntk.Trainer(z, (loss, error), learner, [printer])

    samplesPerSeq   = 1000
    sequences       = 1792

    validSeqs       = 200

    minibatchSize   = batchSize * samplesPerSeq
    minibatches     = sequences // batchSize
    validBatches    = validSeqs // batchSize

    print("Input sequence length: {} days; Total Sequences: {}".format(samplesPerSeq, sequences + validSeqs))
    cntk.logging.log_number_of_parameters(z)
    print("{} epochs; {} minibatches per epoch".format(numEpochs, minibatches))

    for e in range(numEpochs):
        for b in range(minibatches):
            mb = trainReader.next_minibatch(minibatchSize, trainInputMap)
            trainer.train_minibatch(mb)
        trainer.summarize_training_progress()

        # Validate results
        for b in range(validBatches):
            mb = validReader.next_minibatch(minibatchSize, validInputMap)
            trainer.test_minibatch(mb)
        trainer.summarize_test_progress()
        print()



    return

train()