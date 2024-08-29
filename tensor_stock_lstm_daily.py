# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:43:38 2024

@author: MANJEET SINGH


Stock-MArket-Forecasting--Lstm git repo



"""

import yfinance as yf
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy
from datetime import datetime,timedelta
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


time_step=60
n_steps=60
prediction_size=30
# Define the ticker symbol for the stock you're interested in
ticker_symbol = 'AAPL'
# ticker_symbol = 'WMT'
# ticker_symbol = 'TSLA'
# ticker_symbol = 'NFLX'

ticker = yf.Ticker(ticker_symbol)

data=ticker.info
df = ticker.history(period='5y',)
df.reset_index(inplace=True)

df['prevClose'] = df['Close'].shift(1)






df1=df[['Open','prevClose','Date']]

df1.set_index('Date', inplace=True)

plt.plot(df1)



scaler_open=MinMaxScaler(feature_range=(0,1))
df1.iloc[:,0]=scaler_open.fit_transform(np.array(df1.iloc[:,0]).reshape(-1,1))
df1.iloc[:,1]=scaler_open.transform(np.array(df1.iloc[:,1]).reshape(-1,1))








##splitting dataset into train and test split
training_size=int(len(df1)*1)
training_size1=int(len(df1)*0.65)
test_size=len(df1)-training_size1
train_data,test_data=df1.iloc[0:training_size,:],df1.iloc[training_size1:len(df1),:]


# convert an array of values into a dataset matrix


def create_dataset_high(dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset.iloc[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset.iloc[i + time_step, 0])
    dataX=numpy.array(dataX)
    dataY=numpy.array(dataY)
    return dataX, dataY



# time_step = 100
X_train, y_train = create_dataset_high(train_data, time_step)
X_test, ytest = create_dataset_high(test_data, time_step)


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 2)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 2)

huber_loss = tf.keras.losses.Huber()

model_open=Sequential()
model_open.add(Bidirectional(LSTM(50, return_sequences=True,input_shape=(n_steps,2))))
model_open.add(Bidirectional(LSTM(50,return_sequences=True,kernel_regularizer=regularizers.l2(0.01))))
model_open.add(Bidirectional(LSTM(50)))
model_open.add(Dense(5, activation="relu"))
# model.add(Dense(3, activation="relu"))
model_open.add(Dense(1))
model_open.compile(loss=huber_loss,optimizer=Adam(learning_rate=0.001))

model_open.summary()


model_open.fit(X_train,y_train,validation_data=(X_train,y_train),epochs=25,batch_size=40,verbose=1)





#############################################################








# train_predict=model_open.predict(X_train)
# test_predict=model_open.predict(X_test)

# ##Transformback to original form
# train_predict=scaler_open.inverse_transform(train_predict)
# test_predict=scaler_open.inverse_transform(test_predict)

# ### Calculate RMSE performance metrics
# import math
# from sklearn.metrics import mean_squared_error
# math.sqrt(mean_squared_error(y_train,train_predict))

# ### Test Data RMSE
# math.sqrt(mean_squared_error(ytest,test_predict))


### Plotting 
# shift train predictions for plotting
# look_back=140
# trainPredictPlot = numpy.empty_like(df1)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(df1)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# # plot baseline and predictions                      3490-2364
# plt.plot(scaler_open.inverse_transform(df1))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()



len(test_data)

prevclose=test_data.iloc[-prediction_size:,1]
x_input=test_data[341:]

x_input.shape

x_input=np.array(x_input)

temp_input=np.array(x_input)
# temp_input=temp_input[0].tolist()

temp_input

from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<prediction_size):
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
       
        x_input=x_input
        x_input = x_input.reshape((1, n_steps, 2))
        #print(x_input)
        yhat = model_open.predict(x_input, verbose=0)
        if i==0:
            prevclose=prevclose[i]
        else:
            prevclose=yhat[0]
        new = np.array([yhat[0], prevclose]).reshape(1, 2)
        
        # Concatenating along axis 0 (rows)
        temp_input = np.concatenate((temp_input, new), axis=0)
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,2))
        yhat = model_open.predict(x_input, verbose=0)
        if i==0:
            prevclose=prevclose[i]
        else:
            prevclose=yhat[0]
        new = np.array([yhat[0], [prevclose]]).reshape(1, 2)
        
        # Concatenating along axis 0 (rows)
        temp_input = np.concatenate((temp_input, new), axis=0)
        # temp_input.append(new)
       
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# day_new=np.arange(1,101)
# day_pred=np.arange(101,101+len(lst_output_high))

import matplotlib.pyplot as plt

len(df1)




# plt.plot(day_new,scaler_open.inverse_transform(df1[3391:]))
# plt.plot(day_pred,scaler_open.inverse_transform(lst_output))

# df3=df1.tolist()
# df3.extend(lst_output)
# plt.plot(df3[3491:])


# df3=scaler_open.inverse_transform(df3).tolist()
# plt.plot(df3)

# plt.plot(scaler_open.inverse_transform(df1).tolist())
# plt.show()




""" 
Model to predict High

"""


df1=df[['High','Open','Date']]

df1.set_index('Date', inplace=True)

plt.plot(df1)



scaler_high=MinMaxScaler(feature_range=(0,1))
df1.iloc[:,0]=scaler_high.fit_transform(np.array(df1.iloc[:,0]).reshape(-1,1))
df1.iloc[:,1]=scaler_high.transform(np.array(df1.iloc[:,1]).reshape(-1,1))








##splitting dataset into train and test split
training_size=int(len(df1)*1)
training_size1=int(len(df1)*0.65)
test_size=len(df1)-training_size1
train_data,test_data=df1.iloc[0:training_size,:],df1.iloc[training_size1:len(df1),:]


# convert an array of values into a dataset matrix


def create_dataset_high(dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset.iloc[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset.iloc[i + time_step, 0])
    dataX=numpy.array(dataX)
    dataY=numpy.array(dataY)
    return dataX, dataY



# time_step = 100
X_train, y_train = create_dataset_high(train_data, time_step)
X_test, ytest = create_dataset_high(test_data, time_step)


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 2)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 2)



model_high=Sequential()
model_high.add(Bidirectional(LSTM(50, return_sequences=True,input_shape=(n_steps,2))))
model_high.add(Bidirectional(LSTM(50,return_sequences=True,kernel_regularizer=regularizers.l2(0.01))))
model_high.add(Bidirectional(LSTM(50)))
model_high.add(Dense(5, activation="relu"))
# model.add(Dense(3, activation="relu"))
model_high.add(Dense(1))
model_high.compile(loss=huber_loss,optimizer=Adam(learning_rate=0.001))

model_high.summary()


model_high.fit(X_train,y_train,validation_data=(X_train,y_train),epochs=25,batch_size=40,verbose=1)




#############################################################





# train_predict=model_high.predict(X_train)
# test_predict=model_high.predict(X_test)

# ##Transformback to original form
# train_predict=scaler_high.inverse_transform(train_predict)
# test_predict=scaler_high.inverse_transform(test_predict)

# ### Calculate RMSE performance metrics
# import math
# from sklearn.metrics import mean_squared_error
# math.sqrt(mean_squared_error(y_train,train_predict))

# ### Test Data RMSE
# math.sqrt(mean_squared_error(ytest,test_predict))


### Plotting 
# shift train predictions for plotting
# look_back=140
# trainPredictPlot = numpy.empty_like(df1)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(df1)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# # plot baseline and predictions
# plt.plot(scaler_high.inverse_transform(df1))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()


len(test_data)


x_input=test_data[341:]
x_input.shape

x_input=np.array(x_input)

temp_input=np.array(x_input)
# temp_input=temp_input[0].tolist()

temp_input

from numpy import array

lst_output_high=[]
n_steps=100
i=0
while(i<prediction_size):
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
       
        x_input=x_input
        x_input = x_input.reshape((1, n_steps, 2))
        #print(x_input)
        yhat = model_high.predict(x_input, verbose=0)
        
        new = np.array([yhat[0], lst_output[i]]).reshape(1, 2)
        
        # Concatenating along axis 0 (rows)
        temp_input = np.concatenate((temp_input, new), axis=0)
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output_high.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,2))
        yhat = model_high.predict(x_input, verbose=0)
       
        new = np.array([yhat[0], lst_output[i]]).reshape(1, 2)
        
        # Concatenating along axis 0 (rows)
        temp_input = np.concatenate((temp_input, new), axis=0)
        # temp_input.append(new)
       
        lst_output_high.extend(yhat.tolist())
        i=i+1
    

print(lst_output_high)


day_new=np.arange(1,101)
day_pred=np.arange(101,101+len(lst_output_high))

import matplotlib.pyplot as plt

len(df1)


# plt.plot(day_new,scaler_high.inverse_transform(df1[3391:]))
# plt.plot(day_pred,scaler_high.inverse_transform(lst_output_high))

# df3=df1.tolist()
# df3.extend(lst_output_high)
# plt.plot(df3[3491:])


# df3=scaler_high.inverse_transform(df3).tolist()
# plt.plot(df3)

# plt.plot(scaler_high.inverse_transform(df1).tolist())
# plt.show()








"""    

Model to predict close

"""



df1=df[['Close']]


plt.plot(df1)



scaler_close=MinMaxScaler(feature_range=(0,1))
df1=scaler_close.fit_transform(np.array(df1).reshape(-1,1))



##splitting dataset into train and test split
training_size=int(len(df1)*1)
training_size1=int(len(df1)*0.65)
test_size=len(df1)-training_size1
train_data,test_data=df1[0:training_size,:],df1[training_size1:len(df1),:1]



def create_dataset_high(dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step,0])
    dataX=numpy.array(dataX)
    dataY=numpy.array(dataY)
    return dataX, dataY



time_step = 100
X_train, y_train = create_dataset_high(train_data, time_step)
X_test, ytest = create_dataset_high(test_data, time_step)


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)



model_close=Sequential()
model_close.add(Bidirectional(LSTM(50,return_sequences=True,input_shape=(100,1))))
model_close.add(Bidirectional(LSTM(50,return_sequences=True,kernel_regularizer=regularizers.l2(0.01))))
model_close.add(Bidirectional(LSTM(50)))
model_close.add(Dense(5, activation="relu"))
# model.add(Dense(3, activation="relu"))
model_close.add(Dense(1))
model_close.compile(loss=huber_loss,optimizer=Adam(learning_rate=0.001))

model_close.summary()


model_close.fit(X_train,y_train,validation_data=(X_train,y_train),epochs=25,batch_size=40,verbose=1)




#############################################################



# train_predict=model_close.predict(X_train)
# test_predict=model_close.predict(X_test)

# ##Transformback to original form
# train_predict=scaler_close.inverse_transform(train_predict)
# test_predict=scaler_close.inverse_transform(test_predict)

# ### Calculate RMSE performance metrics
# import math
# from sklearn.metrics import mean_squared_error
# math.sqrt(mean_squared_error(y_train,train_predict))

# ### Test Data RMSE
# math.sqrt(mean_squared_error(ytest,test_predict))


### Plotting 
# shift train predictions for plotting
# look_back=140
# trainPredictPlot = numpy.empty_like(df1)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(df1)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# # plot baseline and predictions
# plt.plot(scaler_close.inverse_transform(df1))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()


len(test_data)


x_input=test_data[341:].reshape(1,-1)
x_input.shape


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# temp_input=temp_input[0].tolist()

temp_input

from numpy import array

lst_output_close=[]
n_steps=100
i=0
while(i<prediction_size):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
       
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model_close.predict(x_input, verbose=0)
       
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output_close.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model_close.predict(x_input, verbose=0)
       
        temp_input.extend(yhat[0].tolist())
        
        
        lst_output_close.extend(yhat.tolist())
        i=i+1
   

day_new=np.arange(1,101)
day_pred=np.arange(101,101+len(lst_output_high))

import matplotlib.pyplot as plt

len(df1)


# plt.plot(day_new,scaler_close.inverse_transform(df1[3391:]))
# plt.plot(day_pred,scaler_close.inverse_transform(lst_output_low))

# df3=df1.tolist()
# df3.extend(lst_output_high)
# plt.plot(df3[3491:])


# df3=scaler_close.inverse_transform(df3).tolist()
# plt.plot(df3)

# plt.plot(scaler_close.inverse_transform(df1).tolist())
# plt.show()






"""    

Model to predict Low

"""



df1=df[['Low','Close','Date']]

df1.set_index('Date', inplace=True)

plt.plot(df1)



scaler_low=MinMaxScaler(feature_range=(0,1))
df1.iloc[:,0]=scaler_low.fit_transform(np.array(df1.iloc[:,0]).reshape(-1,1))
df1.iloc[:,1]=scaler_low.transform(np.array(df1.iloc[:,1]).reshape(-1,1))








##splitting dataset into train and test split
training_size=int(len(df1)*1)
training_size1=int(len(df1)*0.65)
test_size=len(df1)-training_size1
train_data,test_data=df1.iloc[0:training_size,:],df1.iloc[training_size1:len(df1),:]


# convert an array of values into a dataset matrix


def create_dataset_high(dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset.iloc[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset.iloc[i + time_step, 0])
    dataX=numpy.array(dataX)
    dataY=numpy.array(dataY)
    return dataX, dataY


time_step = 100
X_train, y_train = create_dataset_high(train_data, time_step)
X_test, ytest = create_dataset_high(test_data, time_step)


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 2)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 2)



model_low=Sequential()
model_low.add(Bidirectional(LSTM(50,return_sequences=True,input_shape=(100,2))))
model_low.add(Bidirectional(LSTM(50,return_sequences=True,kernel_regularizer=regularizers.l2(0.01))))
model_low.add(Bidirectional(LSTM(50)))
model_low.add(Dense(5, activation="relu"))
# model.add(Dense(3, activation="relu"))
model_low.add(Dense(1))
model_low.compile(loss=huber_loss,optimizer=Adam(learning_rate=0.001))

model_low.summary()
 
model_low.fit(X_train,y_train,validation_data=(X_train,y_train),epochs=25,batch_size=40,verbose=1)




#############################################################



# train_predict=model_low.predict(X_train)
# test_predict=model_low.predict(X_test)

# ##Transformback to original form
# train_predict=scaler_low.inverse_transform(train_predict)
# test_predict=scaler_low.inverse_transform(test_predict)

# ### Calculate RMSE performance metrics
# import math
# from sklearn.metrics import mean_squared_error
# math.sqrt(mean_squared_error(y_train,train_predict))

# ### Test Data RMSE
# math.sqrt(mean_squared_error(ytest,test_predict))


### Plotting 
# shift train predictions for plotting
# look_back=140
# trainPredictPlot = numpy.empty_like(df1)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(df1)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# # plot baseline and predictions
# plt.plot(scaler_low.inverse_transform(df1))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()


len(test_data)


x_input=test_data[341:]
x_input.shape

x_input=np.array(x_input)

temp_input=np.array(x_input)
# temp_input=temp_input[0].tolist()

temp_input

from numpy import array

lst_output_low=[]
n_steps=100
i=0
while(i<prediction_size):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
       
        x_input=x_input
        x_input = x_input.reshape((1, n_steps, 2))
        #print(x_input)
        yhat = model_low.predict(x_input, verbose=0)
      
        new = np.array([yhat[0], lst_output_close[i]]).reshape(1, 2)
        
        # Concatenating along axis 0 (rows)
        temp_input = np.concatenate((temp_input, new), axis=0)
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output_low.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,2))
        yhat = model_low.predict(x_input, verbose=0)
       
        new = np.array([yhat[0], lst_output_close[i]]).reshape(1, 2)
        
        # Concatenating along axis 0 (rows)
        temp_input = np.concatenate((temp_input, new), axis=0)
        # temp_input.append(new)
       
        lst_output_low.extend(yhat.tolist())
        i=i+1
    

print(lst_output_low)


day_new=np.arange(1,101)
day_pred=np.arange(101,101+len(lst_output_high))

import matplotlib.pyplot as plt

len(df1)


# plt.plot(day_new,scaler_low.inverse_transform(df1[3391:]))
# plt.plot(day_pred,scaler_low.inverse_transform(lst_output_low))

# df3=df1.tolist()
# df3.extend(lst_output_high)
# plt.plot(df3[3491:])


# df3=scaler_low.inverse_transform(df3).tolist()
# plt.plot(df3)

# plt.plot(scaler_low.inverse_transform(df1).tolist())
# plt.show()







############################






dfcandle=pd.DataFrame(index=range(len(lst_output)))


lst_output=scaler_open.inverse_transform(lst_output)
lst_output_high=scaler_high.inverse_transform(lst_output_high)
lst_output_low=scaler_low.inverse_transform(lst_output_low)
lst_output_close= scaler_close.inverse_transform(lst_output_close)



# lst_output= [x[0] for x in lst_output]
# lst_output_high= [x[0] for x in lst_output_high]
# lst_output_low= [x[0] for x in lst_output_low]
# lst_output_close= [x[0] for x in lst_output_close]



dfcandle['Open']=lst_output
dfcandle['High']=lst_output_high
dfcandle['Low']=lst_output_low
dfcandle['Close']=lst_output_close


# Define start and end dates
start_date = datetime.now().date()
end_date = start_date+timedelta(days=50)

# Generate a date range
date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' is for business days
date_range=date_range[:prediction_size]
# Create a list of timestamps for each weekday
timestamps = []
for date in date_range:
    for hour in range(1):  # From 09:00 to 15:00 (inclusive)
        timestamps.append(str(date.strftime(f'%Y-%m-%d 00:00:00-04:00')))


dfcandle['Date']=timestamps
dfcandle['Date']=pd.to_datetime(dfcandle['Date'])



import mplfinance as mpf

# Ensure your dataframe is formatted correctly for mplfinance
dfcandle.set_index('Date', inplace=True)
# df.set_index('Datetime', inplace=True)
dfcandle=dfcandle[-1:]
dfcandle['Open']=
# Create a new DataFrame that contains the required OHLC (Open, High, Low, Close) data
df_candlestick = dfcandle[['Open', 'High', 'Low', 'Close',]]

# Add volume data if you have it
# df_candlestick['Volume'] = df['Volume']


figsize = (12, 8)  # Width x Height in inches
dpi = 100
# Plot the candlestick chart
mpf.plot(df_candlestick, type='candle', style='charles', title='Candlestick Chart',ylabel='Price', volume=False,figsize=figsize)  # volume=True if you have volume data




 # Ensure your dataframe is formatted correctly for mplfinance
# dfcandle.set_index('Date', inplace=True)
# df.set_index('Datetime', inplace=True)


# Create a new DataFrame that contains the required OHLC (Open, High, Low, Close) data
df_candlestick = df[['Open', 'High', 'Low', 'Close','Date']]

# Add volume data if you have it
# df_candlestick['Volume'] = df['Volume']


figsize = (12, 8)  # Width x Height in inches
dpi = 100
# Plot the candlestick chart


df_candlestick.set_index('Date', inplace=True)

df_candlesticks = pd.concat([df_candlestick, dfcandle])

mpf.plot(df_candlestick, type='candle', style='charles', title='Candlestick Chart',ylabel='Price', volume=False,figsize=figsize)  # volume=True if you have volume data





















