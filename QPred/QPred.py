import cntk
import argparse
import numpy as np
from DataReader import DataReader
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
timeShift   = 7     # Number of days ahead to look
timeSteps   = 14    # Number of data points in a sequence
threshold   = 0.03  # % Change we're looking at, up or down
seqDist     = 1     # Distance each sequence is separated from the one before

numFeatures = 6     
numClasses  = 3

numEpochs   = 50
batchSize   = 128

lstmLayers  = 24
lstmSize    = 64    # TODO: Why are we getting NAN loss when lstmSize >= 96

def createReader(filePath, isTraining, inputDim, outputDim):
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(filePath, cntk.io.StreamDefs(
        features = cntk.io.StreamDef(field='X', shape=inputDim,  is_sparse=False),
        labels   = cntk.io.StreamDef(field='Y', shape=outputDim, is_sparse=True),
        )), randomize=isTraining, max_sweeps=cntk.io.INFINITELY_REPEAT)
                                   

def createModel(input, numClasses, layers, lstmLayers):
    model = cntk.layers.Sequential([
        cntk.layers.For(range(layers), lambda: 
                   cntk.layers.Sequential([
                       #cntk.layers.Stabilizer(), 
                       cntk.layers.Recurrence(cntk.layers.LSTM(lstmLayers), go_backwards=False)
                   ])),
        cntk.sequence.last,
        #cntk.layers.Dropout(0.1),
        cntk.layers.Dense(numClasses)
        ])
    return model

# Going to start out trying to predict whether
# a stock will have changed (up or down) more than x% after n days
def train():

    # TODO: Need to add a method that reads exact sample size when
    # we're loading data that's already been converted
    #convertData(dataPath, 'intel', threshold, timeSteps, timeShift, seqDist)

    input   = cntk.sequence.input_variable((numFeatures), name='features')
    label   = cntk.input_variable((numClasses), name='label')

    trainReader = createReader('./data/intel_train.ctf', True,  numFeatures, numClasses)
    validReader = createReader('./data/intel_valid.ctf', False, numFeatures, numClasses)

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
    accy    = cntk.element_not(cntk.classification_error(z, label)) # Print accuracy %, not error! 

    lr          = cntk.learning_parameter_schedule(0.005, batchSize)
    learner     = cntk.adam(z.parameters, lr, 0.9) #, l2_regularization_weight=0.00001, gradient_clipping_threshold_per_sample=5.0
    #tbWriter    = cntk.logging.TensorBoardProgressWriter(1, './Tensorboard/', model=model)
    printer     = cntk.logging.ProgressPrinter(100, tag='Training')
    trainer     = cntk.Trainer(z, (loss, accy), learner, [printer])

    # TODO: These should be automatically detected!
    samplesPerSeq   = timeSteps
    sequences       = 8709
    validSeqs       = 968

    minibatchSize   = batchSize * samplesPerSeq
    minibatches     = sequences // batchSize
    validBatches    = validSeqs // batchSize

    cntk.logging.log_number_of_parameters(z)
    print("Input days: {}; Looking for +- {:.1f}% change {} days ahead;".format(samplesPerSeq, threshold*100.0, timeShift))
    print("Total Sequences: {}; {} epochs; {} minibatches per epoch;".format(sequences + validSeqs, numEpochs, minibatches+validBatches))

    # Testing out custom data reader
    reader = DataReader('./data/intel_train.ctf', numFeatures, numClasses, batchSize, timeSteps, False)
    testReader = DataReader('./data/intel_valid.ctf', numFeatures, numClasses, batchSize, timeSteps, False)
    for e in range(numEpochs):
        # Train network
        for b in range(minibatches):
            X, Y = next(reader)
            trainer.train_minibatch({ z.arguments[0]: X, label: Y })
        trainer.summarize_training_progress()

        for b in range(minibatches):
            X, Y = next(testReader)
            trainer.test_minibatch({ z.arguments[0]: X, label: Y })
        trainer.summarize_test_progress()


    for e in range(numEpochs):
        # Train network
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




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-convert')

    train()

