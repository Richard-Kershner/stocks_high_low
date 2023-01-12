#!/usr/bin/env python
# coding: utf-8

# # First look
# ## 1) Plot data.  Will use IBM for easy reference
# ## 2) Linear Regression on Low
# ### Plot Actual value on Low Value on Low deviation back ground
# ## 3) Linear Regression, Low, Close up
# ## 4) Liner Regression, Low/High
# ## 5) Rolling Values
# ## ---------------------------------------------------------------

# 
# 
# 
# 
# ## 1) Plot data.  Will use IBM for easy reference

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib widget for Jupiter lab...

plt.rcParams['figure.figsize'] = [20, 5]# width, height in inches

from sklearn.linear_model import LinearRegression

df = pd.read_csv('D:/data/stocks/ixtradingChart_5y/ibm.csv', index_col=0)

print("Plot Actual data for IBM Stock High/Low")

df = df.sort_values('date_ordinal')
print(df.tail())

plt.plot(df['date_ordinal'].values, df['low'].values)
plt.plot(df['date_ordinal'].values, df['high'].values)
plt.show()


# ## 2) Linear Regression on Low
# ### Plot Actual value on Low Value on Low deviation back ground

# In[2]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

modelLow = LinearRegression()

# ------ divide data into training and testing
X_train = df.values[:1000]
y_train = df['low'].values[1:1001]
modelLow.fit(X_train, y_train)
X_test = df.values[1000:]
y_test = df['low'].values[1001:]

#------ Predict Low value
predLow = modelLow.predict(X_test)

# ----------- Compute Error of Low value
msError = mean_squared_error(y_test , predLow[:-1])
print('error', msError)
print(predLow[-3:])
print('error %', 100*msError/predLow[-3])
print('error', msError)

# Plot Low value Predicted highlighted in back ground with error
xAxis = df['date_ordinal'].values[1000:]
y1 = predLow + msError
y2 = predLow - msError
plt.fill_between(xAxis, y1, y2, alpha=.3, color='red')
plt.plot(xAxis, predLow, c='red')
plt.plot(xAxis, df['low'].values[1000:], c='black')

plt.show()


# 
# 
# ## 3) Linear Regression, Low, Close up

# In[3]:


# ============ Narrow down and zoom in on last part of data for better view
viewLast = 10
xAxis = df['date_ordinal'].values[-viewLast:]
y1 = predLow[-viewLast:] + msError
y2 = predLow[-viewLast:] - msError
plt.fill_between(xAxis, y1, y2, alpha=.3, color='red')
plt.plot(xAxis, predLow[-viewLast:], c='red', alpha=.7)
plt.plot(xAxis, df['low'].values[-viewLast:], c='black')

plt.plot(xAxis, df['high'].values[-viewLast:], c='black')

plt.show()


# 
# 
# 
# ## 4) Liner Regression, Low/High

# In[4]:


# ========== Show both high/low predictions overlapping

# ---------------- Create and Train model for high values.  Generate Error
modelH = LinearRegression()

XH_train = df.values[:1000]
yH_train = df['high'].values[1:1001]
modelH.fit(XH_train, yH_train)
XH_test = df.values[1000:]
yH_test = df['high'].values[1001:]
predHigh = modelH.predict(XH_test)

msErrorH = mean_squared_error(y_test , predHigh[:-1])

# =====  Using pridiction Low from Above, plot both so can compare
print("Black lines show actual high/low")
print("Blue is predicted High")
print("Red is predicted Low")
print("Red/Blue background is error")

viewLast = 10

plt.rcParams['figure.figsize'] = [20, 10]# width, height in inches
xAxis = df['date_ordinal'].values[-viewLast:] # 

# ----- fill error for Low in red
y1 = predLow[-viewLast:] + msError
y2 = predLow[-viewLast:] - msError
plt.fill_between(xAxis, y1, y2, alpha=.3, color='red')

# ----- fill error for high in blue
y1 = predHigh[-viewLast:] + msErrorH
y2 = predHigh[-viewLast:] - msErrorH
plt.fill_between(xAxis, y1, y2, alpha=.3, color='blue')

# -------- plot predicted values in  red and blue
plt.plot(xAxis, predLow[-viewLast:], c='red') # , alpha=.7
plt.plot(xAxis, df['high'].values[-viewLast:], c='blue')

# ------- plot actual values in black
plt.plot(xAxis, df['low'].values[-viewLast:], c='black')
plt.plot(xAxis, predHigh[-viewLast:], c='black') # , alpha=.7

plt.show()


# ## 5) Rolling Values
# ### One common practiceis to average values over a period, looking at long term predictions

# In[5]:


# working with rolling window of data.

# reload because redoes calculations
df = pd.read_csv('D:/data/stocks/ixtradingChart_5y/ibm.csv', index_col=0)
df = df.sort_values('date_ordinal')

windowSize = 20
predUpTo = -10 # matches above

print(df.head())
print()

# rolling windows has to be .mean() .median() .max()  .min()

df['winLow']=df['low']
df['winLow']=df['low'].rolling(windowSize).mean()
df['winHigh']=df['high']
df['winHigh']=df['winHigh'].rolling(windowSize).mean()
df = df.dropna()

X_train = df.iloc[:predUpTo-windowSize][['winHigh','winLow','volume']]
y_train_low = df.iloc[windowSize:predUpTo]['winLow']
y_train_high = df.iloc[windowSize:predUpTo]['winHigh']
print(np.shape(X_train), np.shape(y_train_low), np.shape(y_train_high))

X_test = df.iloc[predUpTo-windowSize:-windowSize][['winHigh','winLow','volume']]
y_test_low = df.iloc[predUpTo:]['winLow']
y_test_high = df.iloc[predUpTo:]['winHigh']
print(np.shape(X_test), np.shape(y_test_low), np.shape(y_test_high))
print()

modelLow = LinearRegression()
modelHigh = LinearRegression()

modelLow.fit(X_train.values, y_train_low)
modelHigh.fit(X_train.values, y_train_high)

predLow = modelLow.predict(X_test.values)
predHigh = modelHigh.predict(X_test.values)

msErrorL = mean_squared_error(y_test_low , predLow)
msErrorH = mean_squared_error(y_test_high , predHigh)
print("=========")

viewLast = -predUpTo # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

plt.rcParams['figure.figsize'] = [20, 10]# width, height in inches
xAxis = df.iloc[predUpTo:]['date_ordinal'].values 

# ----- fill error for Low in red
y1 = predLow[-viewLast:] + msError
y2 = predLow[-viewLast:] - msError
plt.fill_between(xAxis, y1, y2, alpha=.3, color='red')

# ----- fill error for high in blue
y1 = predHigh[-viewLast:] + msErrorH
y2 = predHigh[-viewLast:] - msErrorH
plt.fill_between(xAxis, y1, y2, alpha=.3, color='blue')

# -------- plot predicted values in  red and blue
plt.plot(xAxis, predLow, c='red') # , alpha=.7
plt.plot(xAxis, predHigh, c='blue')

# ------- plot actual values in black
plt.plot(xAxis, y_test_low, c='black')
plt.plot(xAxis, y_test_high, c='black')

plt.show()


# In[6]:


# how far into the future can we go???????

