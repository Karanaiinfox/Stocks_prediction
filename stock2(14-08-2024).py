# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:59:23 2024

@author: Karan Rana
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
from sklearn.decomposition import PCA

# Define the ticker symbol for the stock you're interested in
ticker_symbol = 'AAPL'
ticker_symbol = 'WMT' 
ticker_symbol = 'SHY'  # Example: Apple Inc.

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

data=ticker.info

df = ticker.history(period='2y', interval='1h')
df.reset_index(inplace=True)

df['Close'] = df['Close']*83.93
df['Low'] = df['Low']*83.93
df['High'] = df['High']*83.93
df['Open'] = df['Open']*83.93
df['Volume'] = df['Volume']

df['Prev1_Close'] = df['Close'].shift(1)
df['Prev1_Low'] = df['Low'].shift(1)
df['Prev1_High'] = df['High'].shift(1)
df['Prev1_Open'] = df['Open'].shift(1)
df['Prev1_Volume'] = df['Volume'].shift(1)


df['Prev2_Close'] = df['Close'].shift(2)
df['Prev2_Low'] = df['Low'].shift(2)
df['Prev2_High'] = df['High'].shift(2)
df['Prev2_Open'] = df['Open'].shift(2)
df['Prev2_Volume'] = df['Volume'].shift(2)


df['Prev3_Close'] = df['Close'].shift(3)
df['Prev3_Low'] = df['Low'].shift(3)
df['Prev3_High'] = df['High'].shift(3)
df['Prev3_Open'] = df['Open'].shift(3)
df['Prev3_Volume'] = df['Volume'].shift(3)

df['Prev4_Close'] = df['Close'].shift(4)
df['Prev4_Low'] = df['Low'].shift(4)
df['Prev4_High'] = df['High'].shift(4)
df['Prev4_Open'] = df['Open'].shift(4)
df['Prev4_Volume'] = df['Volume'].shift(4)

df['Prev5_Close'] = df['Close'].shift(5)
df['Prev5_Low'] = df['Low'].shift(5)
df['Prev5_High'] = df['High'].shift(5)
df['Prev5_Open'] = df['Open'].shift(5)
df['Prev5_Volume'] = df['Volume'].shift(5)

df['Prev6_Close'] = df['Close'].shift(6)
df['Prev6_Low'] = df['Low'].shift(6)
df['Prev6_High'] = df['High'].shift(6)
df['Prev6_Open'] = df['Open'].shift(6)
df['Prev6_Volume'] = df['Volume'].shift(6)



df['Prev7_Close'] = df['Close'].shift(7)
df['Prev7_Low'] = df['Low'].shift(7)
df['Prev7_High'] = df['High'].shift(7)
df['Prev7_Open'] = df['Open'].shift(7)
df['Prev7_Volume'] = df['Volume'].shift(7)



df['Prev8_Close'] = df['Close'].shift(8)
df['Prev8_Low'] = df['Low'].shift(8)
df['Prev8_High'] = df['High'].shift(8)
df['Prev8_Open'] = df['Open'].shift(8)
df['Prev8_Volume'] = df['Volume'].shift(8)


df['Prev9_Close'] = df['Close'].shift(9)
df['Prev9_Low'] = df['Low'].shift(9)
df['Prev9_High'] = df['High'].shift(9)
df['Prev9_Open'] = df['Open'].shift(9)
df['Prev9_Volume'] = df['Volume'].shift(9)

df['Prev10_Close'] = df['Close'].shift(10)
df['Prev10_Low'] = df['Low'].shift(10)
df['Prev10_High'] = df['High'].shift(10)
df['Prev10_Open'] = df['Open'].shift(10)
df['Prev10_Volume'] = df['Volume'].shift(10)


df['Prev11_Close'] = df['Close'].shift(11)
df['Prev11_Low'] = df['Low'].shift(11)
df['Prev11_High'] = df['High'].shift(11)
df['Prev11_Open'] = df['Open'].shift(11)
df['Prev11_Volume'] = df['Volume'].shift(11)


df['Prev12_Close'] = df['Close'].shift(12)
df['Prev12_Low'] = df['Low'].shift(12)
df['Prev12_High'] = df['High'].shift(12)
df['Prev12_Open'] = df['Open'].shift(12)
df['Prev12_Volume'] = df['Volume'].shift(12)


df['Prev13_Close'] = df['Close'].shift(13)
df['Prev13_Low'] = df['Low'].shift(13)
df['Prev13_High'] = df['High'].shift(13)
df['Prev13_Open'] = df['Open'].shift(13)
df['Prev13_Volume'] = df['Volume'].shift(13)

df['Prev14_Close'] = df['Close'].shift(14)
df['Prev14_Low'] = df['Low'].shift(14)
df['Prev14_High'] = df['High'].shift(14)
df['Prev14_Open'] = df['Open'].shift(14)
df['Prev14_Volume'] = df['Volume'].shift(14)

df['Prev15_Close'] = df['Close'].shift(15)
df['Prev15_Low'] = df['Low'].shift(15)
df['Prev15_High'] = df['High'].shift(15)
df['Prev15_Open'] = df['Open'].shift(15)
df['Prev15_Volume'] = df['Volume'].shift(15)

df['Prev16_Close'] = df['Close'].shift(16)
df['Prev16_Low'] = df['Low'].shift(16)
df['Prev16_High'] = df['High'].shift(16)
df['Prev16_Open'] = df['Open'].shift(16)
df['Prev16_Volume'] = df['Volume'].shift(16)



df['Prev17_Close'] = df['Close'].shift(17)
df['Prev17_Low'] = df['Low'].shift(17)
df['Prev17_High'] = df['High'].shift(17)
df['Prev17_Open'] = df['Open'].shift(17)
df['Prev17_Volume'] = df['Volume'].shift(17)



df['Prev18_Close'] = df['Close'].shift(18)
df['Prev18_Low'] = df['Low'].shift(18)
df['Prev18_High'] = df['High'].shift(18)
df['Prev18_Open'] = df['Open'].shift(18)
df['Prev18_Volume'] = df['Volume'].shift(18)

 
df['Prev19_Close'] = df['Close'].shift(19)
df['Prev19_Low'] = df['Low'].shift(19)
df['Prev19_High'] = df['High'].shift(19)
df['Prev19_Open'] = df['Open'].shift(19)
df['Prev19_Volume'] = df['Volume'].shift(19)

df['Prev20_Close'] = df['Close'].shift(20)
df['Prev20_Low'] = df['Low'].shift(20)
df['Prev20_High'] = df['High'].shift(20)
df['Prev20_Open'] = df['Open'].shift(20)
df['Prev20_Volume'] = df['Volume'].shift(20)



# df = pd.read_csv('HistoricalData_1723524781400.csv')
df['Datetime']=pd.to_datetime(df['Datetime'])
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["week_day"] = df["Datetime"].dt.weekday
df["year"] = df["Datetime"].dt.year
df["quarter"] = df["Datetime"].dt.quarter
df["month"] = df["Datetime"].dt.month
df = df.dropna()
y= df['Open']


x = df.drop(columns=['Datetime','Dividends','Close','Open','High','Low','Stock Splits','Volume'])

correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()

le = LabelEncoder()

# Fit and transform the categorical data
x['year'] = le.fit_transform(x['year'])

scaler_for_open = StandardScaler()

x = scaler_for_open.fit_transform(x)

# Number of components you want to keep

# Fit and transform the data
# x = pca.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.09,random_state=4 )


# model = LinearRegression()
# model = DecisionTreeRegressor(criterion='squared_error',)

param_grid = {
    'n_estimators': [100, 200 ,500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10,],
    'min_samples_leaf': [1, 2, 4,6,8,10],
    'bootstrap': [True, False]
}


model_open = RandomForestRegressor(random_state=5)
model_open.fit(X_train, y_train)



# Fit the model on the training data

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                            cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')


# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
   
# Predict on the test data using the best model
# y_pred = best_model.predict(X_test)
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


x = df.drop(columns=['Datetime','Dividends','Close','High','Low','Stock Splits','Volume'])

correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()



# Fit and transform the categorical data
x['year'] = le.fit_transform(x['year'])

scaler_for_high = StandardScaler()

x = scaler_for_high.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01,random_state=4 )


# model = LinearRegression()
# model = DecisionTreeRegressor(criterion='squared_error',)



model_high = RandomForestRegressor(random_state=5)
model_high.fit(X_train, y_train)
# Fit the model on the training data

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                            cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')


# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
   
# Predict on the test data using the best model
# y_pred = best_model.predict(X_test)
y_pred = model_high.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)




print("\nMean Squared Error:", mse)
print("R^2 Score:", r2*100)





"""

model for predicting low


"""


y= df['Low']


x = df.drop(columns=['Datetime','Dividends','Close','Low','Stock Splits','Volume'])

correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()



# Fit and transform the categorical data
x['year'] = le.fit_transform(x['year'])

scaler_for_low = StandardScaler()

x = scaler_for_low.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=4 )


# model = LinearRegression()
# model = DecisionTreeRegressor(criterion='squared_error',)



model_low = RandomForestRegressor(random_state=5)
model_low.fit(X_train, y_train)
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

model for predicting close


"""


y= df['Close']


x = df.drop(columns=['Datetime','Dividends','Close','Stock Splits','Volume'])

correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
# df=df[["Close"]].copy()
# df.describe()


# Fit and transform the categorical data
x['year'] = le.fit_transform(x['year'])

scaler_for_close = StandardScaler()

x = scaler_for_close.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=4 )


# model = LinearRegression()
# model = DecisionTreeRegressor(criterion='squared_error',)



model_close = RandomForestRegressor(random_state=5)
model_close.fit(X_train, y_train)
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
import pandas as pd
from datetime import datetime, timedelta

# # Get the current date
# current_date = datetime.now().date()

# # Define the start and end times
# start_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=9, minutes=30)
# end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=15, minutes=30)

# # Generate a range of datetimes
# time_range = pd.date_range(start=start_time, end=end_time, freq='30T')

# # Convert to DataFrame
# df = pd.DataFrame(time_range, columns=['Datetime'])

# # Adjust timezone to match the provided format (e.g., -04:00)
# df['Datetime'] = df['Datetime'].dt.tz_localize('America/New_York').dt.tz_localize(None)  # Localize to Eastern Time and remove tz info for display

# Print the DataFrame

selected_date=datetime.now()-timedelta(days=2)

df_new = ticker.history(period='5d', interval="1h")
df_new.reset_index(inplace=True)


df_new['Close'] =df_new['Close']*83.93
df_new['Low'] =df_new['Low']*83.93
df_new['High'] =df_new['High']*83.93 
df_new['Open'] =df_new['Open']*83.93


df_new['Prev1_Close'] =df_new['Close'].shift(1)
df_new['Prev1_Low'] =df_new['Low'].shift(1)
df_new['Prev1_High'] =df_new['High'].shift(1)
df_new['Prev1_Open'] =df_new['Open'].shift(1)
df_new['Prev1_Volume'] =df_new['Volume'].shift(1)


df_new['Prev2_Close'] =df_new['Close'].shift(2)
df_new['Prev2_Low'] =df_new['Low'].shift(2)
df_new['Prev2_High'] =df_new['High'].shift(2)
df_new['Prev2_Open'] =df_new['Open'].shift(2)
df_new['Prev2_Volume'] =df_new['Volume'].shift(2)


df_new['Prev3_Close'] = df_new['Close'].shift(3)
df_new['Prev3_Low'] = df_new['Low'].shift(3)
df_new['Prev3_High'] = df_new['High'].shift(3)
df_new['Prev3_Open'] = df_new['Open'].shift(3)
df_new['Prev3_Volume'] = df_new['Volume'].shift(3)

df_new['Prev4_Close'] = df_new['Close'].shift(4)
df_new['Prev4_Low'] = df_new['Low'].shift(4)
df_new['Prev4_High'] = df_new['High'].shift(4)
df_new['Prev4_Open'] = df_new['Open'].shift(4)
df_new['Prev4_Volume'] = df_new['Volume'].shift(4)

df_new['Prev5_Close'] = df_new['Close'].shift(5)
df_new['Prev5_Low'] = df_new['Low'].shift(5)
df_new['Prev5_High'] = df_new['High'].shift(5)
df_new['Prev5_Open'] = df_new['Open'].shift(5)
df_new['Prev5_Volume'] = df_new['Volume'].shift(5)

df_new['Prev6_Close'] = df_new['Close'].shift(6)
df_new['Prev6_Low'] = df_new['Low'].shift(6)
df_new['Prev6_High'] = df_new['High'].shift(6)
df_new['Prev6_Open'] = df_new['Open'].shift(6)
df_new['Prev6_Volume'] = df_new['Volume'].shift(6)



df_new['Prev7_Close'] = df_new['Close'].shift(7)
df_new['Prev7_Low'] = df_new['Low'].shift(7)
df_new['Prev7_High'] = df_new['High'].shift(7)
df_new['Prev7_Open'] = df_new['Open'].shift(7)
df_new['Prev7_Volume'] = df_new['Volume'].shift(7)



df_new['Prev8_Close'] = df_new['Close'].shift(8)
df_new['Prev8_Low'] = df_new['Low'].shift(8)
df_new['Prev8_High'] = df_new['High'].shift(8)
df_new['Prev8_Open'] = df_new['Open'].shift(8)
df_new['Prev8_Volume'] = df_new['Volume'].shift(8)


df_new['Prev9_Close'] = df_new['Close'].shift(9)
df_new['Prev9_Low'] = df_new['Low'].shift(9)
df_new['Prev9_High'] = df_new['High'].shift(9)
df_new['Prev9_Open'] = df_new['Open'].shift(9)
df_new['Prev9_Volume'] = df_new['Volume'].shift(9)

df_new['Prev10_Close'] = df_new['Close'].shift(10)
df_new['Prev10_Low'] = df_new['Low'].shift(10)
df_new['Prev10_High'] = df_new['High'].shift(10)
df_new['Prev10_Open'] = df_new['Open'].shift(10)
df_new['Prev10_Volume'] = df_new['Volume'].shift(10)


df_new['Prev11_Close'] = df_new['Close'].shift(11)
df_new['Prev11_Low'] = df_new['Low'].shift(11)
df_new['Prev11_High'] = df_new['High'].shift(11)
df_new['Prev11_Open'] = df_new['Open'].shift(11)
df_new['Prev11_Volume'] = df_new['Volume'].shift(11)


df_new['Prev12_Close'] = df_new['Close'].shift(12)
df_new['Prev12_Low'] = df_new['Low'].shift(12)
df_new['Prev12_High'] = df_new['High'].shift(12)
df_new['Prev12_Open'] = df_new['Open'].shift(12)
df_new['Prev12_Volume'] = df_new['Volume'].shift(12)


df_new['Prev13_Close'] = df_new['Close'].shift(13)
df_new['Prev13_Low'] = df_new['Low'].shift(13)
df_new['Prev13_High'] = df_new['High'].shift(13)
df_new['Prev13_Open'] = df_new['Open'].shift(13)
df_new['Prev13_Volume'] = df_new['Volume'].shift(13)

df_new['Prev14_Close'] = df_new['Close'].shift(14)
df_new['Prev14_Low'] = df_new['Low'].shift(14)
df_new['Prev14_High'] = df_new['High'].shift(14)
df_new['Prev14_Open'] = df_new['Open'].shift(14)
df_new['Prev14_Volume'] = df_new['Volume'].shift(14)

df_new['Prev15_Close'] = df_new['Close'].shift(15)
df_new['Prev15_Low'] = df_new['Low'].shift(15)
df_new['Prev15_High'] = df_new['High'].shift(15)
df_new['Prev15_Open'] = df_new['Open'].shift(15)
df_new['Prev15_Volume'] = df_new['Volume'].shift(15)

df_new['Prev16_Close'] = df_new['Close'].shift(16)
df_new['Prev16_Low'] = df_new['Low'].shift(16)
df_new['Prev16_High'] = df_new['High'].shift(16)
df_new['Prev16_Open'] = df_new['Open'].shift(16)
df_new['Prev16_Volume'] = df_new['Volume'].shift(16)



df_new['Prev17_Close'] = df_new['Close'].shift(17)
df_new['Prev17_Low'] = df_new['Low'].shift(17)
df_new['Prev17_High'] = df_new['High'].shift(17)
df_new['Prev17_Open'] = df_new['Open'].shift(17)
df_new['Prev17_Volume'] = df_new['Volume'].shift(17)



df_new['Prev18_Close'] = df_new['Close'].shift(18)
df_new['Prev18_Low'] = df_new['Low'].shift(18)
df_new['Prev18_High'] = df_new['High'].shift(18)
df_new['Prev18_Open'] = df_new['Open'].shift(18)
df_new['Prev18_Volume'] = df_new['Volume'].shift(18)

 
df_new['Prev19_Close'] = df_new['Close'].shift(19)
df_new['Prev19_Low'] = df_new['Low'].shift(19)
df_new['Prev19_High'] = df_new['High'].shift(19)
df_new['Prev19_Open'] = df_new['Open'].shift(19)
df_new['Prev19_Volume'] = df_new['Volume'].shift(19)

df_new['Prev20_Close'] = df_new['Close'].shift(20)
df_new['Prev20_Low'] = df_new['Low'].shift(20)
df_new['Prev20_High'] = df_new['High'].shift(20)
df_new['Prev20_Open'] = df_new['Open'].shift(20)
df_new['Prev20_Volume'] = df_new['Volume'].shift(20)

# df = pd.read_csv('HistoricalData_1723524781400.csv')
df_new['Datetime']=pd.to_datetime(df_new['Datetime'])
df_new["hour"] =df_new["Datetime"].dt.hour
df_new["day"] =df_new["Datetime"].dt.day
df_new["week_day"] =df_new["Datetime"].dt.weekday
df_new["year"] =df_new["Datetime"].dt.year
df_new["quarter"] =df_new["Datetime"].dt.quarter
df_new["month"] =df_new["Datetime"].dt.month
df_new =df_new.dropna()


df_new_original=df_new[df_new['Datetime'].dt.date == selected_date.date()]
df_new_original.reset_index(drop=True, inplace=True)
df_filtered_all =df_new[df_new['Datetime'].dt.date == selected_date.date()]

df_filtered=df_filtered_all
df_filtered_all.reset_index(drop=True, inplace=True)
for i in range(len(df_filtered_all)-1):
    print(i)
    df_filtered = pd.DataFrame([df_new_original.iloc[i]])
    df_filtered['year'] = le.transform(df_filtered['year'])
    
    # predicting open
    
    xforpred = df_filtered.drop(columns=['Datetime','Dividends','Close','Open','High','Low','Stock Splits','Volume'])  
    xforpred = scaler_for_open.transform(xforpred)
    open_today=model_open.predict(xforpred)
    
    
    # predicting high
    df_filtered['Open']=open_today
    xforpred = df_filtered.drop(columns=['Datetime','Dividends','Close','High','Low','Stock Splits','Volume'])  
    xforpred = scaler_for_high.transform(xforpred)
    high_today=model_high.predict(xforpred)
    
    
    
    # predicting low
    df_filtered['High']=high_today
    xforpred = df_filtered.drop(columns=['Datetime','Dividends','Close','Low','Stock Splits','Volume'])  
    xforpred = scaler_for_low.transform(xforpred)
    low_today=model_low.predict(xforpred)
    
    
    # predicting close
    df_filtered['Low']=low_today
    xforpred = df_filtered.drop(columns=['Datetime','Dividends','Close','Stock Splits','Volume'])  
    xforpred = scaler_for_close.transform(xforpred)
    close_today=model_close.predict(xforpred)
    
    df_filtered['close']=close_today
    
    
    
    df_filtered_all.loc[i] = df_filtered.iloc[0]



df_new_original.info()

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

# Plot Open
axs[0].plot(df_filtered_all['Datetime'], df_filtered_all['Open'], label='Predicted - Open', marker='o')
axs[0].plot(df_new_original['Datetime'],df_new_original['Open'], label='Actual - Open', marker='x')
axs[0].set_ylabel('Open')
axs[0].legend()
axs[0].grid(True)

# Plot High
axs[1].plot(df_filtered_all['Datetime'], df_filtered_all['High'], label='Filtered All - High', marker='o')
axs[1].plot(df_new_original['Datetime'],df_new_original['High'], label='Actual - High', marker='x')
axs[1].set_ylabel('High')
axs[1].legend()
axs[1].grid(True)

# Plot Low
axs[2].plot(df_filtered_all['Datetime'], df_filtered_all['Low'], label='Predicted - Low', marker='o')
axs[2].plot(df_new_original['Datetime'],df_new_original['Low'], label='Actual - Low', marker='x')
axs[2].set_ylabel('Low')
axs[2].legend()
axs[2].grid(True)

# Plot Close
axs[3].plot(df_filtered_all['Datetime'], df_filtered_all['Close'], label='Predicted - Close', marker='o')
axs[3].plot(df_new_original['Datetime'],df_new_original['Close'], label='Actual - Close', marker='x')
axs[3].set_xlabel('Datetime')
axs[3].set_ylabel('Close')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()





