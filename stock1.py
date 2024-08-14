# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:33:23 2024

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

# Define the ticker symbol for the stock you're interested in
ticker_symbol = 'AAPL'
ticker_symbol = 'WMT'  # Example: Apple Inc.

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

data=ticker.info
df = ticker.history(period='2y', interval='1h')
df.reset_index(inplace=True)


df['Prev_Close'] = df['Close']*100
df['Prev_Low'] = df['Low']*100
df['Prev_High'] = df['High']*100
df['Prev_Open'] = df['Open']*100

df['Prev_Close'] = df['Close'].shift(1)
df['Prev_Low'] = df['Low'].shift(1)
df['Prev_High'] = df['High'].shift(1)
df['Prev_Open'] = df['Open'].shift(1)
df['Prev_Volume'] = df['Volume'].shift(1)
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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=4 )


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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=4 )


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

selected_date=datetime.now()-timedelta(days=4)
df_new = ticker.history(period='5d', interval="1h")
df_new.reset_index(inplace=True)


df_new['Prev_Close'] = df_new['Close']*100
df_new['Prev_Low'] = df_new['Low']*100
df_new['Prev_High'] = df_new['High']*100
df_new['Prev_Open'] = df_new['Open']*100

df_new['Prev_Close'] = df_new['Close'].shift(1)
df_new['Prev_Low'] = df_new['Low'].shift(1)
df_new['Prev_High'] = df_new['High'].shift(1)
df_new['Prev_Open'] = df_new['Open'].shift(1)
df_new['Prev_Volume'] = df_new['Volume'].shift(1)
# df = pd.read_csv('HistoricalData_1723524781400.csv')
df_new['Datetime']=pd.to_datetime(df_new['Datetime'])
df_new["hour"] = df_new["Datetime"].dt.hour
df_new["day"] = df_new["Datetime"].dt.day
df_new["week_day"] = df_new["Datetime"].dt.weekday
df_new["year"] = df_new["Datetime"].dt.year
df_new["quarter"] = df_new["Datetime"].dt.quarter
df_new["month"] = df_new["Datetime"].dt.month
df_new = df_new.dropna()


df_new_original=df_new[df_new['Datetime'].dt.date == selected_date.date()]
df_new_original.reset_index(drop=True, inplace=True)
df_filtered_all = df_new[df_new['Datetime'].dt.date == selected_date.date()]

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
axs[0].plot(df_new_original['Datetime'], df_new_original['Open'], label='Actual - Open', marker='x')
axs[0].set_ylabel('Open')
axs[0].legend()
axs[0].grid(True)

# Plot High
axs[1].plot(df_filtered_all['Datetime'], df_filtered_all['High'], label='Filtered All - High', marker='o')
axs[1].plot(df_new_original['Datetime'], df_new_original['High'], label='Actual - High', marker='x')
axs[1].set_ylabel('High')
axs[1].legend()
axs[1].grid(True)

# Plot Low
axs[2].plot(df_filtered_all['Datetime'], df_filtered_all['Low'], label='Predicted - Low', marker='o')
axs[2].plot(df_new_original['Datetime'], df_new_original['Low'], label='Actual - Low', marker='x')
axs[2].set_ylabel('Low')
axs[2].legend()
axs[2].grid(True)

# Plot Close
axs[3].plot(df_filtered_all['Datetime'], df_filtered_all['Close'], label='Predicted - Close', marker='o')
axs[3].plot(df_new_original['Datetime'], df_new_original['Close'], label='Actual - Close', marker='x')
axs[3].set_xlabel('Datetime')
axs[3].set_ylabel('Close')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()





