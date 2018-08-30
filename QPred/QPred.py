import cntk
import numpy as np
import matplotlib.pyplot as plt

# Pandas datareader error workaround 
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pddr

# TODO: Look into online data readers with pd 
#tickers = ['APPL']
#startDate = '2010-01-01'
#endDate = '2016-12-31'
#panelData = pddr.DataReader('INPX', 'google', startDate, endDate)

def createModel(input, outDim, hiddenDim):
    return

# Going to start out trying to predict whether
# a stock will have changed (up or down) more than x% after n days
def train():

    features    = cntk.sequence.input_variable((1), name='features')
    label       = cntk.input_variable((3), name='label')

    model   = createModel()

    loss    = cntk.cross_entropy_with_softmax(model, label)
    error   = cntk.classification_error(model, label)

    return