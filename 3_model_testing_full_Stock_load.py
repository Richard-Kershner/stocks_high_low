#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

import numpy as np
import pandas as pd
import itertools

import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

#import tensorflow as tf
from sklearn import linear_model
from sklearn import svm  #support vector machine
from sklearn import neighbors # nearet neighbor
from sklearn import tree # decision tree

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
#import seaborn as sns

import math as math
import sqlite3
from datetime import datetime

# One of the questions if the data is looked at like a photograph of time....
# Would this help in predictability
# The end result still shows no.  A variation not tested here is using Tensor Flow convoluted neural network
def reformat_2D_series(df, defs): # must be in order from earliest to latest
    # defs = {col:[col1, col2], histCount:4}
    if 'col' not in defs or 'histCount' not in defs:
        return df # must have all the info to process
    df2 = df[defs['col']]
    for i in range(1,defs['histCount']):
        for col in defs['col']:
            df2[col + '_' + str(i)]=df[col].shift(i)
    npA = df2.dropna().values
    d1 = int(np.shape(npA)[0])
    d2 = int(np.shape(npA)[1]/defs['histCount'])
    npA = npA.reshape(d1, defs['histCount'], d2)
    return npA

# -----------  scaling, with exception of the neural network, has little effect on models
class noScale():
    def __init__(self):
        pass
    def transform(self, X):
        return X
    def fit(self, X):pass
    
class scaling():
    def __init__(self):
        self.currentModel = ''
        self.currentArgs = {}
        self.models = {}
        self.models = {'noScale':{'model':noScale},
                       'StandardScaler':{'model':StandardScaler}}
    def fit_transform(self, X,xAll):
        if self.currentModel != '':
            scaler = self.models[self.currentModel]['model']()
            #print('60 scaler', scaler)
            scaleTest = StandardScaler()
            scaleTest.fit(X)
            scaler.fit(X)
            xAll = scaler.transform(xAll)
        return xAll
    def getModelList(self):
        scaleList = []
        for key in self.models:
            scaleList.append(key)
        return scaleList
    
# Iterates through a model list with different args.
# Fits the training data
# Returns prediction on for testing data
class model_testing:
    def __init__(self):
        self.currentModel = ''
        self.currentArgs = {}
        self.models = {}
    def set_model(self, modelName, args):
        self.currentModel = modelName
        self.currentArgs = args
    def create_model(self):
        if self.currentModel in self.models:
            if self.currentArgs == {}:
                model = self.models[self.currentModel]['model']()
            else:
                model = self.models[self.currentModel]['model'](**self.currentArgs)
            return model
        print("something went wrong")
        return None     
    def train_predict(self, X_train, y_train, X_test):
        model = None
        if self.currentModel in self.models:
            if self.currentArgs == {}:
                model = self.models[self.currentModel]['model']()
                #.fit_predict(X_train, y_train, X_test)
                #return y_pred
            else: # my_function(**data)
                model = self.models[self.currentModel]['model'](**self.currentArgs)
        if model != None:
            model.fit(X_train, y_train)
            pred_y = model.predict(X_test)
            return pred_y  
        return None
    def getModelList(self):
        modelKeys = []
        for key in self.models:
            if 'args' not in self.models[key]: 
                modelKeys.append([key])
            else:
                args = []
                products = []
                for a in self.models[key]['args']:
                    products.append(self.models[key]['args'][a])
                    args.append(a)
                fullList=list(itertools.product(*products))
                for row in fullList:#valueAlpha, valueCV....
                    actualArgs = {}
                    for pnt in range(len(args)):
                        actualArgs[args[pnt]] = row[pnt]
                    modelKeys.append([key, actualArgs])
        return modelKeys
    
# Create Linear Regression models to be used in testing
class linear_regression_models(model_testing):
    def __init__(self):
        model_testing.__init__(self)
        self.models = {'LinearRegression':{'model':linear_model.LinearRegression},
                      'Ridge':{'model':linear_model.Ridge,
                               'args':{'alpha':[.0005,.005,.05,.1,.5]}},
                      'RidgeCV':{'model':linear_model.RidgeCV,
                                 'args':{'alphas':[(0.1, 1.0, 10.0),(0.01, 0.1, 1.0),(1.0, 10.0, 20.0)],
                                        'cv':[2,3,4]}},
                       'Lasso':{'model':linear_model.Lasso,
                               'args':{'alpha':[.05,.5,1,1.5,2]}},
                      'LassoLars':{'model':linear_model.LassoLars,
                                'args':{'alpha':[.05,.5,1,1.5,2]}}
                      }
        
# Create other regression models to be using in testing        
class regression_models(model_testing):
    def __init__(self):
        model_testing.__init__(self)
        self.models = {'SVR':{'model':svm.SVR},
                      'KNeighborsRegressor':{'model':neighbors.KNeighborsRegressor},
                           # n_neighbors : int, optional (default = 5)
                           # weights uniform, distance
                           # algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
                           # leaf_size : int, optional (default = 30)
                      'DecisionTreeRegressor':{'model':tree.DecisionTreeRegressor},
                           # criterion=’mse’, L2 , friedman_mse, mae
                      'GradientBoostingRegressor':{'model':GradientBoostingRegressor},
                           # loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, optional (default=’ls’)
                           # learning_rate : float, optional (default=0.1)
                           # n_estimators : int (default=100)
                           # criterion (default=”friedman_mse”) mse, mae
                           # max_depth : integer, optional (default=3)
                           # alpha : float (default=0.9)
                      'GaussianProcessRegressor':{'model':GaussianProcessRegressor},
                           # class sklearn.gaussian_process.GaussianProcessRegressor
                           # (kernel=None, alpha=1e-10, optimizer=’fmin_l_bfgs_b’, 
                           # n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
                      'MLPRegressor':{'model':MLPRegressor,
                                      # MLPRegressor(hidden_layer_sizes=(100, ), 
                                      # activation= {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
                                      # solver= {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
                                      # alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, 
                                      # power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
                                      # tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                                      # early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
                                      #n_iter_no_change=10)
                                     'args':{
                                         'hidden_layer_sizes':[(10,),(100,),(10,10),(100,100)],
                                         'max_iter':[200,500,1000]
                                     }}}

# Create different clustering models to be used in testing the prediction.
class clustering_models(model_testing):
    def __init__(self):
        model_testing.__init__(self)
        self.models = {'K-Means':{'model':KMeans,
                                 'args':{'n_clusters':[3,10,50,100,200],
                                        'max_iter':[300, 1000]}},
                      # (n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, 
                       # tol=0.0001, precompute_distances=’auto’, verbose=0, 
                       # random_state=None, copy_x=True, n_jobs=None, algorithm=’auto’)
                      'AffinityPropagation':{'model':AffinityPropagation},
                               # damping=0.5, max_iter=200, 
                               # convergence_iter=15, copy=True, preference=None, 
                               # affinity=’euclidean’, verbose=False
                      'MeanShift':{'model':MeanShift},
                                # bandwidth=None, seeds=None, 
                                # bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None
                      'SpectralClustering':{'model':SpectralClustering,
                                 'args':{'n_clusters':[3,10,50,100,200]}},
                               # (n_clusters=8, eigen_solver=None, 
                               # random_state=None, n_init=10, gamma=1.0, 
                               # affinity=’rbf’, n_neighbors=10, eigen_tol=0.0, 
                               # assign_labels=’kmeans’, degree=3, coef0=1, kernel_params=None, 
                               # n_jobs=None)
                       # Hierarchical clustering  returns a tree instead of cluster's array
                      'DBSCAN':{'model':DBSCAN},
                               # (eps=0.5, min_samples=5, metric=’euclidean’, 
                               # metric_params=None, algorithm=’auto’, 
                               # leaf_size=30, p=None, n_jobs=None)[source]
                       'Birch':{'model':Birch,
                                 'args':{'n_clusters':[3,10,50,100,200]}}
                               # (threshold=0.5, branching_factor=50, 
                               # n_clusters=3, compute_labels=True, copy=True)
                      }                      

# Classification models on this testing where not really helpful, so left out
class classification_models(model_testing):
    def __init__(self):
        model_testing.__init__(self)
        self.models = {'SVR':{'model':svm.SVR}}   

# Loading data from stock file
def loadData(fPath, file):  
    # ============================== load data ==============    
    dataDF = pd.read_csv(fPath + file, index_col=0)
    # print(dataDF.head())
    dataDF['low']=dataDF['low']/dataDF['open']
    dataDF['high']=dataDF['high']/dataDF['open']
    #print(dataDF.head())

    colNames = dataDF.columns
    colPredNames = ['low', 'high']
    colPred = []
    for n in colPredNames:
        colPred.append(dataDF.columns.get_loc(n))
    # dataNP = dataDF.values
    # print("226  load data dataDF.describe()", dataDF.describe())
    # print("227 is null", dataDF.isnull().sum().sum())
    X = dataDF.values[:-1]
    X_last = [dataDF.values[-1]]
    y = dataDF[colPredNames].values[1:]
    # print('233 load data', np.shape(X), np.shape(X_last), np.shape(y))
    # ------------------------------ end load data -------
    return X, X_last, colNames, colPredNames, colPred

def test_DataModel(X, y, colPred, modelClass, pntStart, pntMid,pntShow):
    model = modelClass.create_model()
    
    model.fit(X[:pntMid],y[:pntMid])
    pred = model.predict(X[pntMid:])
    #pred_fut = model.predict([X[-1]])
    pred = np.hstack( (y[pntMid:],pred, y[pntMid:]-pred)  )
    return pred

def test_DataSeries(X, y, colPred, modelClass, pntStart, pntMid,pntShow):
    model = modelClass.create_model()
    model.fit(X[:pntStart], y[:pntStart]) # xtrain, ytrain
    y_pred = model.predict(X[pntStart:pntMid])
    y_train = y[:pntMid]
    #errors = y[pntStart+1:pntMid+1] - y_pred
    X_train = X[:pntMid]
    predicts = []
    for pnt in range(pntMid,len(X)): # 1st run on next line error to create the rest of 
        X_train = np.vstack( (X_train, [X[pnt]]) )
        y_train = np.vstack(  (y_train, y[pnt])  )
        model = modelClass.create_model()
        model.fit(X_train[:-1], y_train[:-1])
        y_pred = model.predict([ X_train[-1] ])
        error = y_train[-1] - y_pred[0]
        predicts.append( np.hstack((y_train[-1], y_pred[0], error)) )
    pred = np.array(predicts) 
    return pred

def test_DataSeries_error(X, y, colPred, modelClass, pntStart, pntMid,pntShow):
    model = modelClass.create_model()
    model.fit(X[:pntStart], y[:pntStart]) # xtrain, ytrain
    y_pred = model.predict(X[pntStart:pntMid])
    y_train = y[pntStart:pntMid-1]
    errors = y[pntStart:pntMid] - y_pred
    X_train = np.hstack(( X[pntStart:pntMid-1], errors[:-1] ))
    error = y[pntMid] - y_pred[-1]
    predicts = []
    for pnt in range(pntMid,len(X)): # 1st run on next line error to create the rest of 
        newRow =np.hstack( ([X[pnt]], [error])  )
        X_train = np.vstack( (X_train, newRow) )
        y_train = np.vstack(  (y_train, [y[pnt]])  )
        model = modelClass.create_model()
        model.fit(X_train[:-1], y_train[:-1])
        y_pred = model.predict([ X_train[-1] ])
        error = y_train[-1] - y_pred[0]
        predicts.append( np.hstack((y_train[-1], y_pred[0], error)) )
    pred = np.array(predicts) 
    return pred

mScale = scaling() # fit_transform(self, X,xAll):
mScaleList = mScale.getModelList()
modelsRegLin = linear_regression_models() # train_predict(self, X_train, y_train, X_test)
modelsRegLinList = modelsRegLin.getModelList()
# print(modelsRegLinList)
modelsReg = regression_models()
modelsRegList = modelsReg.getModelList()

testTypes = {'mBase':test_DataModel, 'mSeries':test_DataSeries, 'mSError':test_DataSeries_error}

fPathDB = 'D:/data/stocks/model_testing6.db'
fPath = 'D:/data/stocks/ixtradingChart_5y/'
fileList = os.listdir(fPath)
Q = '''CREATE TABLE IF NOT EXISTS models ( 
 symbol text NOT NULL
 , active integer default 1 
 , model text
 , future integer
 , error text
 , std10 integer
 , std20 integer
 , std30 integer
 , std40 integer
 , std50 integer
 , std60 integer
 , std70 integer
 , std80 integer
 , std90 integer
 , std100 integer
 , PRIMARY KEY (symbol, model)
 )'''
conn = sqlite3.connect(fPathDB)
conn.execute(Q)
conn.close()
for file in fileList[:16]:
    start = datetime.now()
    stockSymb = file.split('.')[0]
    # print(stockSymb, fPath + file)
    # print(mScaleList)
    for scale in mScaleList: 
        mScale.currentModel = scale
        # print(scale)
        # mScale.currentArgs = {}
        X, X_last, colNames, colPredNames, colPred = loadData(fPath, file)
        X = X[~np.isnan(X).any(axis=1)]
        dataLength = len(X)
        pntStart = dataLength - 300
        pntMid = dataLength - 250
        pntEnd = -1
        pntShow = int(.3 * pntMid)
        mScale.fit_transform(X[:pntMid],X)
        for modelName in modelsRegLinList:
            for yIncr in range(1,20): # predict next nextNext...
                y = X[:-yIncr,colPred]
                X_train = X[yIncr:]
                #modelClass = modelsRegLin
                mArgs = {}
                if len(modelName)==2: mArgs=modelName[1]
                modelsRegLin.set_model(modelName[0], mArgs)
                for key in testTypes:
                    fullName = scale+'_'+key+'_'+str(modelName)
                    fullName=fullName.replace(',','_').replace(".0",'').replace(':','-')
                    stripS = "'[]{} . ()"
                    for c in stripS:
                        fullName=fullName.replace(c,'')                    
                    try:
                        warnings.simplefilter('ignore')
                        pred = testTypes[key](X_train, y, colPred, modelsRegLin, pntStart, pntMid,pntShow)
                        std=[]
                        for i in range(10,110,10):
                            std.append(np.std(pred[:-i,2*len(colPred):3*len(colPred)]))   
                        # print(len(pred),len(X_train[-len(pred):,0]), s)                        
                        # Q = r'''select * from tracker;'''
                        # tracker = pd.read_sql_query(Q, conn)
                        # stockSymb, fullName, future= yIncr, std
                        Q = "INSERT OR REPLACE INTO models "
                        # Q += "(symbol, active, model, future, error, std10"
                        # Q += ", std20, std30 , std40, std50, std60"
                        # Q += ", std70, std80, std90, std100)"
                        Q += "VALUES('"
                        Q += stockSymb + "', 1, '" + fullName + "', "
                        Q += str(yIncr) + ", '', "
                        Q += str(std[0]) + ", " # std10 integer
                        Q += str(std[1]) + ", " # std20 integer
                        Q += str(std[2]) + ", " # std30 integer
                        Q += str(std[3]) + ", " # std40 integer
                        Q += str(std[4]) + ", " # std50 integer
                        Q += str(std[5]) + ", " # std60 integer
                        Q += str(std[6]) + ", " # std70 integer
                        Q += str(std[7]) + ", " # std80 integer
                        Q += str(std[8]) + ", " # std90 integer
                        Q += str(std[9]) + ")" # std100 integer

                        conn = sqlite3.connect(fPathDB)
                        cursor = conn.cursor()
                        cursor.execute(Q)
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        Q = "INSERT OR REPLACE INTO models (symbol, active, model, future, error) VALUES('"
                        Q += stockSymb + "', 0, '" + fullName + "', "
                        Q += str(yIncr) + ", '"+str(e)+ "')" # std100 integer
                        conn = sqlite3.connect(fPathDB)
                        cursor = conn.cursor()
                        cursor.execute(Q)
                        conn.commit()
                        conn.close()
    dTm = datetime.now() - start
    print(file, round(dTm.seconds/60), dTm.seconds, dTm.microseconds, end = '   ')
                    
print('---done---')


# In[ ]:


print(fPathDB)
conn = sqlite3.connect(fPathDB)
cursor = conn.cursor()
# Q = "INSERT OR REPLACE INTO models VALUES ('b', 1, 'b', 3, 'c', 4, 'd', 5, 6, 7, 8, 9, 12, 11, 11)"
# cursor.execute(Q)
# conn.commit()
# conn.close()
cursor = conn.cursor()
Q = r'''SELECT * FROM models'''
# print(cursor.execute(r'''SELECT * FROM models''').fetchall() )                      
tracker = pd.read_sql_query(Q, conn)
print('-----')
print(tracker.head())
print('-----')
Q = """SELECT * FROM models ORDER BY std10 DESC limit 100"""
Q = """SELECT *, COUNT(symbol) FROM models 
    GROUP BY symbol HAVING COUNT(symbol)>1 
    ORDER BY std10 ASC limit 100"""
#SELECT column, COUNT(column) FROM table GROUP BY column HAVING COUNT(column) > 1
tracker = pd.read_sql_query(Q, conn)
print(tracker)
conn.close()


# In[ ]:


print(fPathDB)
conn = sqlite3.connect(fPathDB)
cursor = conn.cursor()
Q = """SELECT model,AVG(std50) AS Average FROM models 
    GROUP BY model HAVING COUNT(model)>1 
    ORDER BY Average ASC"""
                      
tracker = pd.read_sql_query(Q, conn)
print(tracker.head(90))


# In[ ]:




