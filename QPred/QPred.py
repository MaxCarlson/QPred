import cntk
import numpy as np
import matplotlib.pyplot as plt

# Pandas datareader error workaround 
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import webr

# Looking into an online data reader. 
tickers = ['APPL']
startDate = '2010-01-01'
endDate = '2016-12-31'
panelData = webr.DataReader('INPX', 'google', startDate, endDate)

def train():
    return