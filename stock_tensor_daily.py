# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:03:32 2024

@author: windows 10
"""

# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

# data_source = 'alphavantage' # alphavantage or kaggle

# if data_source == 'alphavantage':
#     # ====================== Loading Data from Alpha Vantage ==================================

#     api_key = 'AOD0P3CIUHOXPTPP'

#     # American Airlines stock market prices
#     ticker = "AAL"

#     # JSON file with all the stock market data for AAL from the last 20 years
#     url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

#     # Save data to this file
#     file_to_save = 'stock_market_data-%s.csv'%ticker

#     # If you haven't already saved data,
#     # Go ahead and grab the data from the url
#     # And store date, low, high, volume, close, open values to a Pandas DataFrame
#     if not os.path.exists(file_to_save):
#         with urllib.request.urlopen(url_string) as url:
#             data = json.loads(url.read().decode())
#             # extract stock market data
#             data = data['Time Series (Daily)']
#             df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
#             for k,v in data.items():
#                 date = dt.datetime.strptime(k, '%Y-%m-%d')
#                 data_row = [date.date(),float(v['3. low']),float(v['2. high']),
#                             float(v['4. close']),float(v['1. open'])]
#                 df.loc[-1,:] = data_row
#                 df.index = df.index + 1
#         print('Data saved to : %s'%file_to_save)        
#         df.to_csv(file_to_save)

#     # If the data is already there, just load it from the CSV
#     else:
#         print('File already exists. Loading data from CSV')
#         df = pd.read_csv(file_to_save)

# else:

#     # ====================== Loading Data from Kaggle ==================================
#     # You will be using HP's data. Feel free to experiment with other data.
#     # But while doing so, be careful to have a large enough dataset and also pay attention to the data normalization
#     df = pd.read_csv(os.path.join('Stocks','hpq.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])
#     print('Loaded data from the Kaggle repository')

ticker_symbol = 'AAPL'


ticker = yf.Ticker(ticker_symbol)

data=ticker.info

df = ticker.history(period='2y', interval='1h')

df.reset_index(inplace=True)
# Sort DataFrame by date
df = df.sort_values('Datetime')
df['Datetime']=pd.to_datetime(df['Datetime'])

# Double check the result
df.head()

# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
# plt.xticks(range(0,df.shape[0],500),df['Datetime'].loc[::500],rotation=45)
# plt.xlabel('Datetime',fontsize=18)
# plt.ylabel('Mid Price',fontsize=18)
# plt.show()




import mplfinance as mpf

# Ensure your dataframe is formatted correctly for mplfinance
df.set_index('Datetime', inplace=True)
# df.set_index('Datetime', inplace=True)

# Create a new DataFrame that contains the required OHLC (Open, High, Low, Close) data
df_candlestick = df[['Open', 'High', 'Low', 'Close']]

# Add volume data if you have it
df_candlestick['Volume'] = df['Volume']

# Plot the candlestick chart
mpf.plot(df_candlestick, type='candle', style='charles', title='Candlestick Chart',
         ylabel='Price', volume=True)  # volume=True if you have volume data





# First calculate the mid prices from the highest and lowest
high_prices = df.loc[:,'High'].values
low_prices = df.loc[:,'Low'].values
mid_prices = (high_prices + low_prices) / 2.0
 



train_data = mid_prices[:3000]
test_data = mid_prices[3000:]



# Scale the data to be between 0 and 1
# When scaling remember! You normalize both test and train data with respect to training data
# Because you are not supposed to have access to test data
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)


# Train the Scaler with training data and smooth data
smoothing_window_size = 500
for di in range(0,2500,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])


# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)


# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(3000):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)


window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Datetime']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))






plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Datetime')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()



window_size = 100
N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))







plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Datetime')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Data Generator
class DataGeneratorSeq:
    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]

            self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)
        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._prices_length - 1))

# LSTM Model Definition
D = 1  # Dimensionality of the data
num_unrollings = 50  # Number of time steps
batch_size = 500  # Number of samples in a batch
num_nodes = [200, 200, 150]  # Hidden nodes in each LSTM layer
dropout = 0.2  # Dropout rate

inputs = layers.Input(shape=(num_unrollings, D))
x = inputs

# LSTM layers with dropout
for i in range(len(num_nodes)):
    return_sequences = i < len(num_nodes) - 1
    x = layers.LSTM(num_nodes[i], return_sequences=return_sequences, dropout=dropout)(x)

outputs = layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Training and Validation
epochs = 30
n_predict_once = 50  # Number of prediction steps
train_mse_ot = []  # Store train losses
test_mse_ot = []  # Store test losses
predictions_over_time = []  # Store predictions

train_seq_length = len(train_data)
data_gen = DataGeneratorSeq(train_data, batch_size, num_unrollings)
test_points_seq = np.arange(11000, 12000, 50).tolist()

for ep in range(epochs):
    average_loss = 0
    for step in range(train_seq_length // batch_size):
        u_data, u_labels = data_gen.unroll_batches()
        x_train = np.stack(u_data, axis=1).reshape((batch_size, num_unrollings, D))
        y_train = np.stack(u_labels, axis=1).reshape((batch_size, num_unrollings, 1))
        
        loss = model.train_on_batch(x_train, y_train)
        average_loss += loss

    train_mse_ot.append(average_loss / (train_seq_length // batch_size))
    print(f'Epoch {ep+1}/{epochs}, Loss: {average_loss / (train_seq_length // batch_size):.6f}')

    predictions_seq = []
    mse_test_loss_seq = []

    for w_i in test_points_seq:
        current_price = all_mid_data[w_i-1:w_i + n_predict_once-1]
        predicted_prices = []

        for pred_i in range(n_predict_once):
            predicted_price = model.predict(current_price.reshape(1, num_unrollings, D))
            predicted_prices.append(predicted_price[0, -1, 0])
            current_price = np.roll(current_price, -1)
            current_price[-1] = predicted_price

        mse_test_loss = np.mean((predicted_prices - all_mid_data[w_i:w_i + n_predict_once])**2)
        mse_test_loss_seq.append(mse_test_loss)
        predictions_seq.append(predicted_prices)

    test_mse_ot.append(np.mean(mse_test_loss_seq))
    predictions_over_time.append(predictions_seq)

    print(f'\tTest MSE: {np.mean(mse_test_loss_seq):.6f}')

# Plotting the predictions over time
best_prediction_epoch = np.argmin(test_mse_ot)
plt.figure(figsize=(18, 18))
plt.subplot(2, 1, 1)
plt.plot(range(len(all_mid_data)), all_mid_data, color='b')

start_alpha = 0.25
alpha = np.linspace(start_alpha, 1.0, len(predictions_over_time[::3]))

for p_i, p in enumerate(predictions_over_time[::3]):
    for xval, yval in zip(test_points_seq, p):
        plt.plot(range(xval, xval + n_predict_once), yval, color='r', alpha=alpha[p_i])

plt.title('Evolution of Test Predictions Over Time', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.xlim(11000, 12500)

plt.subplot(2, 1, 2)
plt.plot(range(len(all_mid_data)), all_mid_data, color='b')
for xval, yval in zip(test_points_seq, predictions_over_time[best_prediction_epoch]):
    plt.plot(range(xval, xval + n_predict_once), yval, color='r')

plt.title('Best Test Predictions Over Time', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.xlim(11000, 12500)
plt.show()








