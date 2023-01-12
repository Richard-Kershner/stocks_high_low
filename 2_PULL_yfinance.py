#!/usr/bin/env python
# coding: utf-8

# In[2]:



import os
import pandas as pd
import pandas_datareader # had to install with pip
import datetime

# note, this is the 3rd version of pull.  
#    the first version used quandle which was closed down and disconintued
#    the second version used iextrading which is limitted unless you pay

# As a free source, it is great for playing with data.  For production or use in algotraiding, it is highly
#   Recomended by many tutorials to use a paid service that is more reliable.
#   As of publishing this, yfinance is still the only free pull

# As of first writing this script in 2016 to running the script in 2023, Errors have changed
#   Limits on number of pulls of day granularity have not been addressed.

# as of 1/11/2023 -- Does not run in python 3.10, 3.9.7
#    Is working in Python 3.7

import yfinance as yf
        
class stockPull_yfinance():
    # from viewing a single pull, the following format is what comes out
    #      Date	Open	High	Low	Close	Volume	Dividends	Stock Splits
    #      note default loads Date as index.... 
    mapCols = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stocksplits']
    
    trackName = 'NONE'
    trackDB = pd.DataFrame()
    def __init__(self, stockTP):
        self.stockPath = stockTP
        self.errors = dict()

    def pullAllStocks(self):
        self.errors = dict()
        try:
            # ------  pulls ALL the symbols from nasdaq ------  as index. 
            #     There is additional information on the stock which should be looked at for possible
            #          possible 'features' that can be used in prediction models
            dfStocks = pandas_datareader.nasdaq_trader.get_nasdaq_symbols(retry_count=3, timeout=30, pause=None)
            symbols = dfStocks.index.values
            
            print("pull values from nasdaq.  Note only the stock symbol, index, is used for pulling data.")
            print()
            print(dfStocks.head())
            print()
            print("====================== Pulling individual stocks ============================")
            print()
            
            # ----------------- for testing, this can be limitted with symbols[4:7] etc. so all 9000+ downloads aren't triggered
            for symb in symbols: # --- cleaning the symbols to match correctly.  Note error printout, not all are pulling
                s = symb.replace('^','-').split(".")[0]
                s = s.replace('~','-')
                s = s.replace('$','-') # quandl
                s.strip()
                self.pullStockSymb(s)
        except Exception as e:
            print("error pulling stock list, NO stocks updated", e)
        return self.errors
    
    def pullStockSymb(self, symb):
        
        try:
            fileFullPath = self.stockPath + symb + ".csv"
            # ---------  if exists, open, pull new data, add to dataframe and save
            if os.path.exists(fileFullPath):
                dfH = pd.read_csv(fileFullPath, index_col=0)
                d = datetime.datetime.strptime(dfH.index[-1], '%Y-%m-%d').date()+  datetime.timedelta(days=-10)
                start = d.strftime('%Y-%m-%d')
                # pull only current data on   progress=False to yf.download() suppresses progress bar
                dfH2 = yf.download(symb, start=d, progress=False)#, end="2017-04-30")
                if len(dfH2.index) >= 1: # only make changes if there is new data
                    dfH2 = dfH2.round(2)
                    dfH2.index = dfH2.index.strftime('%Y-%m-%d')
                    dfH = pd.concat([dfH, dfH2])#.index.drop_duplicates()#.reset_index(drop=True)
                    dfH = dfH.loc[~dfH.index.duplicated(keep='last')] 
                    dfH.to_csv(fileFullPath, index=True)
            else:
                # ----- new stock symbol, pull all
                stockPull = yf.Ticker(symb) # suppress progress bar doesn't work on full pull
                dfH = yf.download(symb)
                dfH = dfH.round(2)
                dfH.to_csv(fileFullPath, index=True)
            # print(symb, end=",  ") 
            
        except Exception as e:
            # print(symb, end="_err,  ")
            self.errors[symb]=e

stockSave = 'D:/data/stocks/stocksIndividual_yfinanceRaw/'
print("File and path saving stock into:", stockSave)
print("     note automatically created.")
print()

# create stock pull object
spc = stockPull_yfinance(stockSave)  

#spc.pullStockSymb('AA')  # --------------------------- pulling a single stock
errorsPull = spc.pullAllStocks() # returns spc.errors
print()
print('error cnt', len(spc.errors))
print()
for symb, err in errorsPull.items():
    print(symb, " ", err)
print()    
print('======done==============')


# In[ ]:




