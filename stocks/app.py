from flask import Flask, render_template, jsonify,request
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import warnings
import yfinance as yf
import joblib
import mplfinance as mpf




warnings.filterwarnings('ignore')

app = Flask(__name__)

def get_data(start_date,end_date,history,ticker_symbol):
    model_open=joblib.load(f'saved_models/model_open_{ticker_symbol}.joblib')
    model_low=joblib.load(f'saved_models/model_low_{ticker_symbol}.joblib')
    model_high=joblib.load(f'saved_models/model_high_{ticker_symbol}.joblib')
    model_close=joblib.load(f'saved_models/model_close_{ticker_symbol}.joblib')
    scaler_for_open=joblib.load(f'saved_models/scaler_for_open_{ticker_symbol}.joblib')
    scaler_for_high=joblib.load(f'saved_models/scaler_for_high_{ticker_symbol}.joblib')
    scaler_for_low=joblib.load(f'saved_models/scaler_for_low_{ticker_symbol}.joblib')
    scaler_for_close=joblib.load(f'saved_models/scaler_for_close_{ticker_symbol}.joblib')
    feature=60
    if ticker_symbol == 'MSFT':
        feature=2
    ticker_symbol = ticker_symbol  # Example: Apple Inc.
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='10y')
    df.reset_index(inplace=True)
    currentdate=str(datetime.now().date())
    if currentdate in str(df.iloc[-1]['Date']):
        print('date present+++++++++++++++++++++++++++++++++++')
        df=df[:-1]
    df["day"] = df["Date"].dt.day
    df["week_day"] = df["Date"].dt.weekday
    if ticker_symbol != 'MSFT':
        df["year"] = df["Date"].dt.year
        df["quarter"] = df["Date"].dt.quarter
        df["month"] = df["Date"].dt.month
    df = df.dropna()
    start_date =   datetime.strptime(start_date,'%Y-%m-%d')
    end_date =  datetime.strptime(end_date,'%Y-%m-%d')
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' is for business days
    date_range=date_range[:10]
    timestamps = []
    for date in date_range:
        for hour in range(1):  # From 09:00 to 15:00 (inclusive)
            timestamps.append(str(date.strftime(f'%Y-%m-%d 00:00:00-04:00')))
    df_all = ticker.history(period='5y', )
    df_final=pd.DataFrame()
    df_final['Date']=timestamps
    df_final['Open'] = np.nan
    df_final["High"] = np.nan
    df_final['Low'] = np.nan
    df_final['Close'] = np.nan
    df_final['Volume'] = np.nan
    df_final['Date']=pd.to_datetime(df_final['Date'])
    # df["hour"] = df["Datetime"].dt.hour
    
    for shift in range(1, feature):
        df_final[f'Prev{shift}_Close'] = np.nan
        df_final[f'Prev{shift}_Low'] = np.nan
        df_final[f'Prev{shift}_High'] = np.nan
        df_final[f'Prev{shift}_Open'] = np.nan
        
    df_final['Date']=pd.to_datetime(df_final['Date'])
    df_final["day"] = df_final["Date"].dt.day
    df_final["week_day"] = df_final["Date"].dt.weekday
    if ticker_symbol != 'MSFT':
        df_final["year"] = df_final["Date"].dt.year
        df_final["quarter"] = df_final["Date"].dt.quarter
        df_final["month"] = df_final["Date"].dt.month
    for i in range(len(df_final)):
        for shift in range(1, feature):
            df_final.at[i, f'Prev{shift}_Close'] = df_all.iloc[-(shift)]['Close']
            df_final.at[i, f'Prev{shift}_Low'] = df_all.iloc[-(shift)]['Low']
            df_final.at[i, f'Prev{shift}_High'] = df_all.iloc[-(shift)]['High']
            df_final.at[i, f'Prev{shift}_Open'] = df_all.iloc[-(shift)][ 'Open']
            
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
        x = df_for_pred.drop(columns=['Date','Close','Volume'])
        x = scaler_for_close.transform(x)
        y_pred = model_close.predict(x)
        df_final.at[i, 'Close']=y_pred
        df_all.reset_index(inplace=True)
        df_all=pd.concat([df_all,df_final.loc[[i],['Date','Open','High','Low','Close']]])
        df_all['Date']=pd.to_datetime(df_all['Date'],utc=True)
        df_all.set_index('Date', inplace=True)   
    dfcandle = df_final[['Open', 'High', 'Low', 'Close','Date']]
    dfcandle_past = df[['Open', 'High', 'Low', 'Close','Date']]
    dfcandle_past=dfcandle_past[-history:]
    df_candlesticks = pd.concat([dfcandle_past, dfcandle])
    df_candlesticks['Date']=pd.to_datetime(df_candlesticks['Date'],utc=True)
    
    return df_candlesticks

def get_stock_price(start_date,end_date,history,ticker_symbol):
    print(history,'historyhistory========================')
    model_open=joblib.load(f'saved_models/model_open_{ticker_symbol}.joblib')
    model_low=joblib.load(f'saved_models/model_low_{ticker_symbol}.joblib')
    model_high=joblib.load(f'saved_models/model_high_{ticker_symbol}.joblib')
    model_change=joblib.load(f'saved_models/model_change_{ticker_symbol}.joblib')
    model_close=joblib.load(f'saved_models/model_close_{ticker_symbol}.joblib')
    scaler_for_open=joblib.load(f'saved_models/scaler_for_open_{ticker_symbol}.joblib')
    scaler_for_high=joblib.load(f'saved_models/scaler_for_high_{ticker_symbol}.joblib')
    scaler_for_low=joblib.load(f'saved_models/scaler_for_low_{ticker_symbol}.joblib')
    scaler_for_change=joblib.load(f'saved_models/scaler_for_change_{ticker_symbol}.joblib')
    scaler_for_close=joblib.load(f'saved_models/scaler_for_close_{ticker_symbol}.joblib')
    feature=2
    ticker_symbol = ticker_symbol  # Example: Apple Inc.
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='10y')
    
    df.reset_index(inplace=True)
    currentdate=str(datetime.now().date())

    if currentdate in str(df.iloc[-1]['Date']):
        print('date present+++++++++++++++++++++++++++++++++++')
        df=df[:-1]
    df['Low_Open'] = (df['Low'] - df['Open']) / df['Open']
    df['Low_Open']=df['Low_Open'].shift(1)

    df['variation']=(df['High']-df['Low'])/df['Open']
    
    df['variation']=df['variation'].shift(1)

    df['change']=df['Close'].pct_change() * 100
    
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



    
    df['Date']=pd.to_datetime(df['Date'])
    
    df["day"] = df["Date"].dt.day
    df["week_day"] = df["Date"].dt.weekday
    
    df = df.dropna()
    y= df['change']

    for shifts in range(1, feature):
        df[f'Prev{shifts}_Close'] = df['Close'].shift(shifts)
        df[f'Prev{shifts}_Low'] = df['Low'].shift(shifts)
        df[f'Prev{shifts}_High'] = df['High'].shift(shifts)
        df[f'Prev{shifts}_Open'] = df['Open'].shift(shifts)
       
    df['Date']=pd.to_datetime(df['Date'])
    df["day"] = df["Date"].dt.day
    df["week_day"] = df["Date"].dt.weekday
    df = df.dropna()
    start_date =   datetime.strptime(start_date,'%Y-%m-%d')
    end_date =  datetime.strptime(end_date,'%Y-%m-%d')
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' is for business days
    date_range=date_range[:10]
    timestamps = []
    for date in date_range:
        for hour in range(1):  # From 09:00 to 15:00 (inclusive)
            timestamps.append(str(date.strftime(f'%Y-%m-%d 00:00:00-04:00')))

    print(timestamps)
    df_all = ticker.history(period='5y', )
    df_all['Low_Open'] = (df_all['Low'] - df_all['Open']) / df_all['Open']
    df_all['variation']=(df_all['High']-df_all['Low'])/df_all['Open']
    df_all['change']=df_all['Close'].pct_change() * 100
    df_all['7-Day SMA'] = df_all['Close'].rolling(window=7).mean()
    df_all['7-Day STD'] = df_all['Close'].rolling(window=7).std()
    df_all['High_Close'] = (df_all['High'] - df_all['Close']) / df_all['Open']
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
    df_final["day"] = df_final["Date"].dt.day
    df_final["week_day"] = df_final["Date"].dt.weekday
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

    print(df_final[['Open', 'High', 'Low', 'Close','Date']])
    dfcandle = df_final[['Open', 'High', 'Low', 'Close','Date']]        
    dfcandle_past = df[['Open', 'High', 'Low', 'Close','Date']]
    dfcandle_past=dfcandle_past[-history:]
    df_candlesticks = pd.concat([dfcandle_past, dfcandle])
    df_candlesticks['Date']=pd.to_datetime(df_candlesticks['Date'],utc=True)
    
    return df_candlesticks



@app.route('/', methods=['GET', 'POST'])
def index():
    start_date=''
    end_date=''
    previous_data=150
    ticker_symbol='MSFT'
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='10y')
    df.reset_index(inplace=True)
    data = df[['Open', 'High', 'Low', 'Close','Date']]
    data=data[-150:]
    if request.method=='POST':
        start_date=request.form.get('start_date')
        end_date=request.form.get('end_date')
        previous_data=int(request.form.get('previous_data'))
        ticker_symbol=request.form.get('ticker_symbol')
        if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
            data=get_stock_price(start_date,end_date,previous_data,ticker_symbol)
        else:    
            data=get_data(start_date,end_date,previous_data,ticker_symbol)
    x= list(data['Date'])
    close= list(data['Close'])
    high= list(data['High'])
    low= list(data['Low'])
    open= list(data['Open'])
    print(start_date, end_date )
    return render_template('new.html', x=x, close=close, high=high, low=low, open=open,start_date=start_date, end_date=end_date,  previous_data=previous_data,ticker_symbol=ticker_symbol)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
