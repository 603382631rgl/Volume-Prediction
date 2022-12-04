# Import all necessary packages
import os
import sys
import re
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import time
from datetime import datetime
from time import strftime
from time import gmtime
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import time as tm

import tensorflow as tf
from keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Flatten, Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input, Concatenate
from keras.models import Model
import itertools

import warnings
warnings.filterwarnings("ignore")


import TAQReaders
from TAQReaders.TAQTradesReader import TAQTradesReader
from TAQReaders.TAQQuotesReader import TAQQuotesReader

def generateDataset(df, timeStep=130, trainRatio=0.95, validation=True, validationRatio=0.05):
    '''
    Return a list of training and testing dataset and validation dataset if validation is True

        Parameters:
                df: dataframe that contains the data
                timeStep: number of time steps for each sample
                trainRatio: ratio of training data
                validation: boolean, whether to split training data into training and validation set
                validationRatio: ratio of validation data
        Returns:
                X_train: training data
                Y_train: training label
                X_test: testing data
                Y_test: testing label
                X_val: validation data
                Y_val: validation label
    '''
    dataX = []
    dataY = []
    for i in range(timeStep, len(df)):
        dataX.append(df.values[i-timeStep:i])
        dataY.append(df.values[i, 0]) #Setting the target feature to be volumn

    def trainValidationTestSplit(dataset, trainRatio=trainRatio, validation=validation, validationRatio=validationRatio):
        def testTrainSplit(df, trainRatio=trainRatio):
            cut_pt = int(trainRatio*len(df))
            train, test = df[0:cut_pt], df[cut_pt:]
            return train, test
        if validation:
            train, test = testTrainSplit(dataset, trainRatio=trainRatio)
            train, validation = testTrainSplit(train, trainRatio=1-validationRatio)
            return train, validation, test
        else:
            train, test = testTrainSplit(dataset, trainRatio=trainRatio)
            return train, test
        
    X, Y = np.array(dataX), np.array(dataY)
    if validation:
        X_train, X_validation, X_test = trainValidationTestSplit(X)
        Y_train, Y_validation, Y_test = trainValidationTestSplit(Y)
        return X_train, X_validation, X_test, Y_train, Y_validation, Y_test
    else:
        X_train, X_test = trainValidationTestSplit(X)
        Y_train, Y_test = trainValidationTestSplit(Y)
        return X_train, X_test, Y_train, Y_test

def getData(dates, dataPath, symbol = 'SPY', displayLog = False, extendHours = False):
    '''
    Return a dataframe that contains the data for the given dates
    
            Parameters:
                    dates: list of dates in string format
                    dataPath: path to the data
                    symbol: ticker symbol
                    displayLog: boolean, whether to display the log
                    extendHours: boolean, whether to include extended hours data
            Returns:
                    df: dataframe that contains the data
    '''
    def readData(path):
        with open(path) as f:
            contents = f.readlines()
        columns = contents[0]
        columnsName = []
        blankIndex = columns.find(' ')
        while(blankIndex>=0):
            columnsName.append(columns[:blankIndex])
            columns = columns[blankIndex+1:]
            blankIndex = columns.find(' ')
        columnsName.append(columns[:blankIndex])
        data = []
        for i in range(1,len(contents)):
            temp = []
            current = contents[i]
            blankIndex = current.find(' ')
            while(blankIndex>=0):
                temp.append(current[:blankIndex])
                current = current[blankIndex+1:]
                blankIndex = current.find(' ')
            temp.append(current[:blankIndex])
            data.append(temp)
        df = pd.DataFrame(data = data, columns = columnsName)
        return df

    nDates = len(dates)
    outputDF = []
    for d in range(nDates):

        if displayLog:
            print('Reading file: ', dataPath+symbol+dates[d])
        path = (dataPath + '/' + '/bars.' + symbol + '.' + dates[d])
        df = readData(path)
        df['date'] = dates[d]
        if extendHours:
            outputDF.append(df)
        else:
            outputDF.append(df[(df.time <= '16:00:00') & (df.time > '09:30:00')])
        if displayLog:
            print('Reading file: ', dataPath+symbol+dates[d], ' done!')
    outputDF = pd.concat(outputDF)
    columns = ['date','time','trade_count','trade_volume','trade_first','trade_high','trade_low','trade_last','vwap']
    outputDF = outputDF.loc[:, columns]
    outputDF = outputDF.set_index(['date','time'])
    columnRenames = {'trade_count':'count','trade_volume':'volume','trade_first':'open','trade_high':'high','trade_low':'low','trade_last':'close','vwap':'vwap'}
    outputDF = outputDF.rename(columns = columnRenames)
    # Change the data type of the columns to float
    outputDF = outputDF.astype(float)
    outputDF['logVolume'] = outputDF[['volume']].applymap(lambda x : np.log(x))
    outputDF['hlDiff'] = outputDF[['high','low']].apply(lambda x : x[0]-x[1], axis = 1)
    return outputDF

def getDates(path):
    '''
    Return a list of dates in string format
        
            Parameters:
                    path: path to the data
            Returns:
                    dates: list of dates in string format
    '''
    dates = []
    for file in os.listdir(path):
        if file.startswith("bars"):
            dates.append(file[-8:])
    dates = sorted(dates)
    return dates

def predictionReport(Y_test, Y_predicted, model = 'LSTM'):
    '''
    Return the prediction report
            
            Parameters:
                    Y_test: testing label
                    Y_predicted: predicted label
                    model: model name
            Returns:
                    None
    ''' 
    if(model == 'LSTM'):
        print('Result from LSTM model with volumn and close price features:')
    else:
        print('Results from '+model+' model')
    MAE = mean_absolute_error(Y_test, Y_predicted)
    print("MAE: "+str(MAE))
    RMAE = np.sqrt(MAE)
    print("RMAE: "+str(RMAE))
    MSE = mean_squared_error(Y_test, Y_predicted)
    print("MSE: "+str(MSE))
    RMSE = np.sqrt(MSE)
    print("RMSE: "+str(RMSE))
    CD = 0
    sign1 = np.sign(np.diff(Y_predicted, axis=0))
    sign2 = np.sign(np.diff(Y_test, axis=0))
    for i in range(len(sign1)):
        if(sign1[i]==sign2[i]):
            CD +=1
    CD = CD/len(sign1)
    print("CD: "+str(CD))


def getRecords(dfs):
    '''
    Return the records from dfs that time is not 16:00:00
            
            Parameters:
                    dfs: the dataframe of the data
            Returns:
                    dfs: the dataframe of the data
    '''
    df = dfs[dfs.index.get_level_values('time') != '16:00:00']
    return df

def kpss_test(series, **kw):
    '''
    KPSS test for stationarity
    Null hypothesis: the process is trend stationary
    Alternative hypothesis: the process is not trend stationary
    '''
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

def getTickerPaths(path):
    '''
    Return the ticker suffix of the given path
        
                Parameters:
                        path: path of the file
                Returns:
                        suffix: ticker suffix
    '''
    files = os.listdir(path)
    # regular expression to get the ticker number from files
    pattern = re.compile(r'\d+')
    tickerNumber = []
    for file in files:
        tickerNumber.append(pattern.findall(file))
    tickerNumber.sort()

    tickerNumber =[number[0] for number in tickerNumber if number != []]
    tickers = []
    for file_number in tickerNumber:
        pathExtended = path+'/out-'+file_number

        subFiles = os.listdir(pathExtended) # This is the ticker name
        tickers.append(subFiles[0])

    fileSuffix = [tickerNumber[i]+'/'+tickers[i] for i in range(len(tickers))]
    filePaths = [path+'/out-'+fileSuffix[i] for i in range(len(fileSuffix))]
    return filePaths

def randomSelectTickers(filePaths, nTickers = 20):
    nDataFrames = nTickers
    dataFrames = []
    np.random.seed(2314)
    randomPaths = np.random.choice(filePaths, nDataFrames, replace = False).tolist()

    for i in range(nDataFrames):
        ticker = re.findall(r'out-\d+/(.*)', randomPaths[i]).pop()
        dates = getDates(randomPaths[i])
        df = getData(dates, randomPaths[i], symbol=ticker, displayLog = False)
        df.name = ticker
        dataFrames.append(df)
    
    def filteringDF(dataFrame):
        # Filter out the dataframes with 20% more zero trade volume
        if (dataFrame[dataFrame.volume == 0].shape[0]/dataFrame.shape[0]) > 0.2:
            return False
        else:
            return True
    dataFrames = list(filter(filteringDF, dataFrames))
    return dataFrames
