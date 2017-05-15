from __future__ import division
import numpy as np
import pandas as pd
import quandl, math, datetime, time
import matplotlib.pyplot as plt
from matplotlib import style
import pylab
style.use('ggplot')
import pickle


# Get the Googl Stock data from quandl
df = quandl.get('WIKI/AAPL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# Make additional Useful features from current data
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_change','Adj. Volume']]
df.fillna(value=-99999,inplace=True)

#Define Label for data.
forecast_col = df['Adj. Close']
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = forecast_col.shift(-forecast_out)

#df.dropna(inplace=True)

# Get X and y for training
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])


#Scale X
from sklearn import preprocessing
X = preprocessing.scale(X)
X_new = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)

y = y[:-forecast_out]


#split data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


#finally train the data
from sklearn.linear_model import LinearRegression
from sklearn import svm


mdl1 = LinearRegression()
mdl1.fit(X,y)


# save a classifier via pickling
with open('linearregression_stock_pred_AAPL.pickle','wb') as f:
    pickle.dump(mdl1,f)

'''
#Load a classifier via pickle
pickle_in = open('linearregression_stock_pred_AAPL.pickle','rb')
mdl1 = pickle.load(pickle_in)
'''


mdl2 = svm.SVR()
mdl2.fit(X,y)

print ('Accuracy of linear regression:',mdl1.score(X_test,y_test))
print ('Accuracy of SVM:',mdl2.score(X_test,y_test))

#  Making new Predictions about future stock prices
forecast = mdl1.predict(X_new)
print forecast

# Graph the forecast
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1 )] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
pylab.show()


