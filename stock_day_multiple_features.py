# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:05:36 2024

@author: MANJEET SINGH
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:59:23 2024

@author: Karan Rana
"""

from sklearn.preprocessing import MinMaxScaler
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
from sklearn.decomposition import PCA
from datetime import datetime,timedelta

import warnings
warnings.filterwarnings('ignore')

feature=2

# Define the ticker symbol for the stock you're interested in

ticker_symbol = 'AAPL'
ticker_symbol = 'WMT'
# ticker_symbol = 'TSLA'
ticker_symbol = 'NFLX'
# ticker_symbol = 'MSFT'

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

data=ticker.info

df = ticker.history(period='max')
df.reset_index(inplace=True)

df=df[:-5]




df['Low_Open'] = (df['Low'] - df['Open']) / df['Open']
df['Low_Open']=df['Low_Open'].shift(1)


df['variation']=(df['High']-df['Low'])/df['Open']
plt.plot(range(len(df['variation'])), df['variation'], color='green', label='variation')
df['variation']=df['variation'].shift(1)


df['change']=df['Close'].pct_change() * 100
plt.plot(range(len(df['change'])), df['change'], color='red', label='change')
df['change']=df['change'].shift(1)


df['7-Day SMA'] = df['Close'].rolling(window=7).mean()
df['7-Day SMA']=df['7-Day SMA'].shift(1)


# Calculate the 7-day Standard Deviation (STD)
df['7-Day STD'] = df['Close'].rolling(window=7).std()
df['7-Day STD']=df['7-Day STD'].shift(1)


# Assuming your DataFrame is named df_all
df['High_Close'] = (df['High'] - df['Close']) / df['Open']
df['High_Close']=df['High_Close'].shift(1)


df['diffrence']=df['Open']-df['Close']

for shifts in range(1, feature):
    df[f'Prev{shifts}_Close'] = df['Close'].shift(shifts)
    df[f'Prev{shifts}_Low'] = df['Low'].shift(shifts)
    df[f'Prev{shifts}_High'] = df['High'].shift(shifts)
    df[f'Prev{shifts}_Open'] = df['Open'].shift(shifts)
    df[f'Prev{shifts}_change'] = df['change'].shift(shifts)
    
    df[f'Prev{shifts}_Low_Open'] = df['Low_Open'].shift(shifts)
    df[f'Prev{shifts}_variation'] = df['variation'].shift(shifts)
    df[f'Prev{shifts}_7-Day SMA'] = df['7-Day SMA'].shift(shifts)
    df[f'Prev{shifts}_7-Day STD'] = df['7-Day STD'].shift(shifts)
    df[f'Prev{shifts}_High_Close'] = df['High_Close'].shift(shifts)
    df[f'Prev{shifts}_diffrence'] = df['diffrence'].shift(shifts)



# ddf = pd.read_csv('HistoricalData_1723524781400.csv')
df['Date']=pd.to_datetime(df['Date'])
# df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Date"].dt.day
df["week_day"] = df["Date"].dt.weekday
# df["year"] = df["Date"].dt.year
# df["quarter"] = df["Date"].dt.quarter
# df["month"] = df["Date"].dt.month
df = df.dropna()
y= df['change']






for shifts in range(1, feature):
    df[f'Prev{shifts}_Close'] = df['Close'].shift(shifts)
    df[f'Prev{shifts}_Low'] = df['Low'].shift(shifts)
    df[f'Prev{shifts}_High'] = df['High'].shift(shifts)
    df[f'Prev{shifts}_Open'] = df['Open'].shift(shifts)
    # df[f'Prev{shifts}_Volume'] = df['Volume'].shift(shifts)



# ddf = pd.read_csv('HistoricalData_1723524781400.csv')
df['Date']=pd.to_datetime(df['Date'])
# df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Date"].dt.day
df["week_day"] = df["Date"].dt.weekday
# df["year"] = df["Date"].dt.year
# df["quarter"] = df["Date"].dt.quarter
# df["month"] = df["Date"].dt.month
df = df.dropna()
y= df['Open']


x = df.drop(columns=['Date','Dividends','Close','Open','High','Low','Stock Splits','Volume'])

correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()

le = LabelEncoder()

# Fit and transform the categorical data
# x['year'] = le.fit_transform(x['year'])

scaler_for_open = StandardScaler()

x = scaler_for_open.fit_transform(x)

# Number of components you want to keep

# Fit and transform the data
# x = pca.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.09,random_state=4 )




# # Parameter grid for GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 500],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4, 6, 8],
#     'bootstrap': [True, False],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'criterion': ['mse', 'mae'],
#     'max_samples': [0.5, 0.7, 0.9, None]
# }

# Initialize the model
# model_open = RandomForestRegressor(random_state=5)

# # Perform grid search
# grid_search = GridSearchCV(estimator=model_open, param_grid=param_grid, 
#                             cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# # Fit the model on the training data
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_model = grid_search.best_estimator_

# # Predict on the test data using the best model
# y_pred = best_model.predict(X_test)
# y_pred = model_open.predict(X_test)







model_open = RandomForestRegressor(random_state=5,criterion='squared_error')
model_open.fit(x, y)


y_pred = model_open.predict(X_test)





# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)






print("\nMean Squared Error:", mse)
print("R^2 Score:", r2*100)



"""
  
model for predicting high
  
"""


y= df['High']


x = df.drop(columns=['Date','Dividends','Close','High','Low','Stock Splits','Volume'])

correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()



# Fit and transform the categorical data
# x['year'] = le.fit_transform(x['year'])

scaler_for_high = StandardScaler()

x = scaler_for_high.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01,random_state=4 )


# model = LinearRegression()
# model = DecisionTreeRegressor(criterion='squared_error',)



model_high = RandomForestRegressor(random_state=5)
model_high.fit(x, y)
# Fit the model on the training data

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                            cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')


# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
   
# Predict on the test data using the best model
# y_pred = best_model.predict(X_test)
y_pred = model_high.predict(X_test)




plt.figure(figsize=(16, 8))
plt.plot(range(len(y_test)), y_test, color='black', label='Test Data')
plt.plot(range(len(y_pred)), y_test, color='black', label='predicted')
# Plot the 20-day forecast


# Adding labels and legend
plt.title('Test Data and 20-Day Forecast')
plt.xlabel('Time -->')
plt.ylabel('price -->')
plt.legend()
plt.show()







# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)




print("\nMean Squared Error:", mse)
print("R^2 Score:", r2*100)





"""

model for predicting low


"""


y= df['Low']


x = df.drop(columns=['Date','Dividends','High','Close','Low','Stock Splits','Volume'])

correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()



# Fit and transform the categorical data
# x['year'] = le.fit_transform(x['year'])

scaler_for_low = StandardScaler()

x = scaler_for_low.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=4 )


# model = LinearRegression()
# model = DecisionTreeRegressor(criterion='squared_error',)



model_low = RandomForestRegressor(random_state=5)
model_low.fit(x, y)
# Fit the model on the training data

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                            cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')


# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
   
# Predict on the test data using the best model
# y_pred = best_model.predict(X_test)
y_pred = model_low.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)




print("\nMean Squared Error:", mse)
print("R^2 Score:", r2*100)





"""

model to predict the change

"""





# df['change']=df['change'].shift(1)



# Assuming your DataFrame is named df_all



# Assuming your DataFrame is named df_all
y= df['diffrence']
    
x = df.drop(columns=['Date','Dividends','Close','Stock Splits','Volume','diffrence','change'])

correlation_matrix = x.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()

le = LabelEncoder()

# Fit and transform the categorical data
# x['year'] = le.fit_transform(x['year'])

scaler_for_change = MinMaxScaler(feature_range=(0,1))

x = scaler_for_change.fit_transform(x)

# Number of components you want to keep

# Fit and transform the data
# x = pca.fit_transform(x)


training_size=int(len(x)*1)
training_size1=int(len(x)*0.99)
test_size=len(x)-training_size1
X_train,X_test=x[0:training_size1],x[training_size1:len(x)]
y_train,y_test=y[0:training_size1],y[training_size1:len(y)]




model_change = RandomForestRegressor(random_state=5,criterion='squared_error')
model_change.fit(x, y)


y_pred = model_change.predict(X_test)






y_test=pd.DataFrame(y_test)

y_test.reset_index(inplace=True)
y_test=y_test.drop(columns=['index'])
# Evaluate the model




mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.plot(y_test[-20:])
plt.show()

plt.plot(y_pred[-20:])
plt.show()




print("\nMean Squared Error:", mse)
print("R^2 Score:", r2*100)















"""

model for predicting close


"""


y= df['Close']


x = df.drop(columns=['Date','Dividends','Close','Stock Splits','Volume'])

correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()


# Fit and transform the categorical data
# x['year'] = le.fit_transform(x['year'])

scaler_for_close = StandardScaler()

x = scaler_for_close.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=4 )


# model = LinearRegression()
# model = DecisionTreeRegressor(criterion='squared_error',)



model_close = RandomForestRegressor(random_state=5)
model_close.fit(x, y)
# Fit the model on the training data

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                            cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')


# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
   
# Predict on the test data using the best model
# y_pred = best_model.predict(X_test)
y_pred = model_close.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)




print("\nMean Squared Error:", mse)
print("R^2 Score:", r2*100)






"""
start prediction

"""







from datetime import datetime
import numpy as np

start_date = datetime.now().date()-timedelta(days=9)
end_date = start_date  +timedelta(days=35)

# Generate a date range


date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' is for business days
date_range=date_range[:5]
# Create a list of timestamps for each weekday
timestamps = []
for date in date_range:
    for hour in range(1):  # From 09:00 to 15:00 (inclusive)
        timestamps.append(str(date.strftime(f'%Y-%m-%d 00:00:00-04:00')))





df_all = ticker.history(period='5y', )

df_all=df_all.iloc[:-6,:]



df_all['Low_Open'] = (df_all['Low'] - df_all['Open']) / df_all['Open']

df_all['variation']=(df_all['High']-df_all['Low'])/df_all['Open']

# df_all['variation']=df_all['variation'].shift(1)


df_all['change']=df_all['Close'].pct_change() * 100

# df_all['change']=df_all['change'].shift(1)


df_all['7-Day SMA'] = df_all['Close'].rolling(window=7).mean()
# df_all['7-Day SMA']=df_all['7-Day SMA'].shift(1)

# Calculate the 7-day Standard Deviation (STD)

df_all['7-Day STD'] = df_all['Close'].rolling(window=7).std()

# df_all['7-Day STD']=df_all['7-Day STD'].shift(1)

df_all['High_Close'] = (df_all['High'] - df_all['Close']) / df_all['Open']
# df_all['High_Close']=df_all['High_Close'].shift(1)
# df_all=df_all.iloc[:,:]

df_all['diffrence']=df_all['Open']-df_all['Close']

df_final=pd.DataFrame()
df_final['Date']=timestamps





df_final['Open'] = np.nan
df_final["High"] = np.nan
df_final['Low'] = np.nan
df_final['Close'] = np.nan
df_final['Low_Open'] = np.nan
df_final['Volume'] = np.nan
df_final['variation'] = np.nan
df_final['change'] = np.nan
df_final['7-Day SMA'] = np.nan
df_final['7-Day STD'] = np.nan
df_final['High_Close'] = np.nan
df_final['diffrence'] = np.nan




i=0
shift=1
import numpy as np


for shift in range(1, feature):
    df_final[f'Prev{shift}_Close'] = np.nan
    df_final[f'Prev{shift}_Low'] = np.nan
    df_final[f'Prev{shift}_High'] = np.nan
    df_final[f'Prev{shift}_Open'] = np.nan
    df_final[f'Prev{shifts}_change'] = np.nan
    
    df_final[f'Prev{shifts}_Low_Open'] = np.nan
    df_final[f'Prev{shift}_variation'] = np.nan
    df_final[f'Prev{shift}_7-Day SMA'] = np.nan
    df_final[f'Prev{shift}_7-Day STD'] = np.nan
    df_final[f'Prev{shift}_High_Close'] = np.nan
    df_final[f'Prev{shift}_diffrence'] = np.nan
    
    
df_final['Date']=pd.to_datetime(df_final['Date'])
# df["hour"] = df["Datetime"].dt.hour
df_final["day"] = df_final["Date"].dt.day
df_final["week_day"] = df_final["Date"].dt.weekday
# df_final["year"] = df_final["Date"].dt.year
# df_final["quarter"] = df_final["Date"].dt.quarter
# df_final["month"] = df_final["Date"].dt.month

   
i=0
shift=1

# for i in range(len(df_final)):
#     for shift in range(1, feature):
#           # Ensure the index is valid
#             df_final.at[i, f'Prev{shift}_Close'] = 11 df_all.iloc[-(shift-i)]['Close']
#             df_final.at[i, f'Prev{shift}_Low'] = df_all.iloc[-(shift-i)]['Low']
#             df_final.at[i, f'Prev{shift}_High'] = df_all.iloc[-(shift-i)]['High']
#             df_final.at[i, f'Prev{shift}_Open'] = df_all.iloc[-(shift-i)][ 'Open']
                     
#             # df_final.at[i, f'Prev{shift}_Volume'] = df_all.iloc[-(shift+i)][ 'Volume']



for i in range(len(df_final)):
    df_final.at[i,'Low_Open']= df_all.iloc[-1]['Low_Open']
    df_final.at[i,'variation']= df_all.iloc[-1]['variation']
    df_final.at[i,'change']= df_all.iloc[-1]['change']
    df_final.at[i,'7-Day SMA']= df_all.iloc[-1]['7-Day SMA']
    df_final.at[i,'7-Day STD']= df_all.iloc[-1]['7-Day STD']
    df_final.at[i,'High_Close']= df_all.iloc[-1]['High_Close']
    df_final.at[i,'diffrence']= df_all.iloc[-1]['diffrence']
    for shift in range(1, feature):
        df_final.at[i, f'Prev{shift}_Close'] = df_all.iloc[-(shift)]['Close']
        df_final.at[i, f'Prev{shift}_Low'] = df_all.iloc[-(shift)]['Low']
        df_final.at[i, f'Prev{shift}_High'] = df_all.iloc[-(shift)]['High']
        df_final.at[i, f'Prev{shift}_Open'] = df_all.iloc[-(shift)][ 'Open']
        df_final.at[i, f'Prev{shift}_change'] = df_all.iloc[-(shift)][ 'change']
        
        df_final.at[i, f'Prev{shift}_Low_Open'] = df_all.iloc[-(shift)][ 'Low_Open']
        df_final.at[i, f'Prev{shift}_variation'] = df_all.iloc[-(shift)][ 'variation']
        df_final.at[i, f'Prev{shift}_7-Day SMA'] = df_all.iloc[-(shift)][ '7-Day SMA']
        df_final.at[i, f'Prev{shift}_7-Day STD'] = df_all.iloc[-(shift)][ '7-Day STD']
        df_final.at[i, f'Prev{shift}_High_Close'] = df_all.iloc[-(shift)][ 'High_Close']
        df_final.at[i, f'Prev{shift}_diffrence'] = df_all.iloc[-(shift)][ 'diffrence']
        
    df_for_pred=df_final.iloc[[i],:]
    x = df_for_pred.drop(columns=['Date','Close','Open','High','Low','Volume'])
    x = scaler_for_open.transform(x)
    y_pred = model_open.predict(x)  
    df_final.at[i, 'Open']=y_pred
    df_for_pred=df_final.iloc[[i],:]
    x = df_for_pred.drop(columns=['Date','Close','High','Low','Volume'])
    x = scaler_for_high.transform(x)
    y_pred = model_high.predict(x)
    df_final.at[i, 'High']=y_pred
    df_for_pred=df_final.iloc[[i],:]
    x = df_for_pred.drop(columns=['Date','High','Close','Low','Volume'])
    x = scaler_for_low.transform(x)
    y_pred = model_low.predict(x)
    df_final.at[i, 'Low']=y_pred
    
    df_for_pred=df_final.iloc[[i],:]
    x = df_for_pred.drop(columns=['Date','Close','Volume','diffrence','change'])
    x = scaler_for_change.transform(x)
    y_pred = model_change.predict(x)
    df_final.at[i, 'diffrence']=y_pred
    
    
    
    
    df_for_pred=df_final.iloc[[i],:]
    x = df_for_pred.drop(columns=['Date','Close','Volume'])
    x = scaler_for_close.transform(x)
    y_pred = model_close.predict(x)
    df_final.at[i, 'Close']=y_pred
    
    df_final.at[i,'variation']=(df_final.iloc[i]['High']-df_final.iloc[i]['Low'])/df_final.iloc[i]['Open']
   
    df_all.reset_index(inplace=True)
    df_final.at[i, 'Low_Open'] =(df_final.iloc[i]['Low'] - df_final.iloc[i]['Open']) / df_final.iloc[i]['Open']
    df_final.at[i,'High_Close'] = (df_final.iloc[i]['High'] - df_final.iloc[i]['Close']) / df_final.iloc[i]['Open']
    df_final.at[i,'diffrence']=df_final.iloc[i]['Open']-df_final.iloc[i]['Close']
    
    df_all=pd.concat([df_all,df_final.loc[[i],['Date','Open','High','Low','Close', 'Low_Open','variation','change','High_Close','diffrence']]])
    df_all['Date']=pd.to_datetime(df_all['Date'],utc=True)
    df_all.set_index('Date', inplace=True)   
    df_all['change']=df_all['Close'].pct_change() * 100
    df_all['7-Day SMA'] = df_all['Close'].rolling(window=7).mean()
    df_all['7-Day STD'] = df_all['Close'].rolling(window=7).std()




# df = pd.read_csv('HistoricalData_1723524781400.csv')
# df_final['Date']=pd.to_datetime(df_final['Date'])
# df_complete=df_final
# x = df_final.drop(columns=['Date','Close','Open','High','Low','Volume'])
# x = scaler_for_open.transform(x)
# y_pred = model_open.predict(x)
# df_complete['Open']=y_pred

# df_high=df_complete
# x = df_high.drop(columns=['Date','Close','High','Low','Volume'])
# x = scaler_for_high.transform(x)
# y_pred = model_high.predict(x)
# df_complete['High']=y_pred

# df_low=df_complete
# x = df_low.drop(columns=['Date','High','Close','Low','Volume'])
# x = scaler_for_low.transform(x)
# y_pred = model_low.predict(x)
# df_complete['Low']=y_pred

# df_close=df_complete
# x = df_close.drop(columns=['Date','Close','Volume'])
# x = scaler_for_close.transform(x)
# y_pred = model_close.predict(x)
# df_complete['Close']=y_pred








dfcandle = df_final[['Open', 'High', 'Low', 'Close','Date']]
dfcandle['Date']=pd.to_datetime(dfcandle['Date'],utc=True)
dfcandle=dfcandle.dropna()
import mplfinance as mpf

# Ensure your dataframe is formatted correctly for mplfinance
dfcandle.set_index('Date', inplace=True)
# df.set_index('Date', inplace=True)


# Create a new DataFrame that contains the required OHLC (Open, High, Low, Close) data
df_candlestick = dfcandle[['Open', 'High', 'Low', 'Close',]]

# Add volume data if you have it
# df_candlestick['Volume'] = df['Volume']


figsize = (24, 18)  # Width x Height in inches
dpi = 100
# Plot the candlestick chart
mpf.plot(df_candlestick, type='candle', style='charles', title='Candlestick Chart',ylabel='Price', volume=False,figsize=figsize)  # volume=True if you have volume data







# plotind Actual data

df_all_actual = ticker.history(period='5y', )
df_all_actual.reset_index(inplace=True)

df_candles = df_all_actual[['Open', 'High', 'Low', 'Close','Date']]

df_candles['Date']=pd.to_datetime(df_candles['Date'],utc=True)
df_candles=df_candles[-6:-1]
import mplfinance as mpf

# Ensure your dataframe is formatted correctly for mplfinance
df_candles.set_index('Date', inplace=True)
# df.set_index('Date', inplace=True)


# Create a new DataFrame that contains the required OHLC (Open, High, Low, Close) data
candlestick = df_candles[['Open', 'High', 'Low', 'Close',]]

# Add volume data if you have it
# df_candlestick['Volume'] = df['Volume']


figsize = (24, 18)  # Width x Height in inches
dpi = 100
# Plot the candlestick chart
mpf.plot(candlestick, type='candle', style='charles', title='Candlestick Chart',ylabel='Price', volume=False,figsize=figsize)  # volume=True if you have volume data










plt.figure(figsize=(16, 8))
plt.plot(range(len(df_candlestick)), df_candlestick['Close'], color='red', label='prediction')
plt.plot(range(len(candlestick)), candlestick['Close'], color='Green', label='Actual')
# Plot the 20-day forecast


# Adding labels and legend
plt.title('close')
plt.xlabel('Time -->')
plt.ylabel('price -->')
plt.legend()
plt.show()






plt.figure(figsize=(16, 8))
plt.plot(range(len(df_candlestick)), df_candlestick['Open'], color='red', label='prediction')
plt.plot(range(len(candlestick)), candlestick['Open'], color='Green', label='Actual')
# Plot the 20-day forecast


# Adding labels and legend
plt.title('Open price')
plt.xlabel('Time -->')
plt.ylabel('price -->')
plt.legend()
plt.show()






# Create a new DataFrame that contains the required OHLC (Open, High, Low, Close) data
df_candlestick = df[['Open', 'High', 'Low', 'Close','Date']]
df_candlestick['Date']=pd.to_datetime(df_candlestick['Date'])
df_candlestick=df_candlestick[-150:]
# Add volume data if you have it
# df_candlestick['Volume'] = df['Volume']

figsize = (24, 18)  # Width x Height in inches
dpi = 100
# Plot the candlestick chart

df_candlestick.set_index('Date', inplace=True)

df_candlesticks = pd.concat([df_candlestick, dfcandle])
df_candlesticks.reset_index(inplace=True)
df_candlesticks['Date']=pd.to_datetime(df_candlesticks['Date'],utc=True)

df_candlesticks.set_index('Date', inplace=True)
mpf.plot(df_candlesticks, type='candle', style='charles', title='Candlestick Chart',ylabel='Price', volume=False,figsize=figsize)  # volume=True if you have volume data


import joblib

joblib.dump(model_open,f'model_open_{ticker_symbol}.joblib')
joblib.dump(model_low,f'model_low_{ticker_symbol}.joblib')
joblib.dump(model_high,f'model_high_{ticker_symbol}.joblib')
joblib.dump(model_change,f'model_change_{ticker_symbol}.joblib')
joblib.dump(model_close,f'model_close_{ticker_symbol}.joblib')


joblib.dump(scaler_for_open,f'scaler_for_open_{ticker_symbol}.joblib')
joblib.dump(scaler_for_high,f'scaler_for_high_{ticker_symbol}.joblib')
joblib.dump(scaler_for_low,f'scaler_for_low_{ticker_symbol}.joblib')
joblib.dump(scaler_for_change,f'scaler_for_change_{ticker_symbol}.joblib')
joblib.dump(scaler_for_close,f'scaler_for_close_{ticker_symbol}.joblib')





