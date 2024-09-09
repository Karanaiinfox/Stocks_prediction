from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings
import yfinance as yf
import joblib
import mplfinance as mpf
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, flash
from flask import Flask, render_template, request, flash, redirect, url_for
from datetime import datetime
# from stock import db
warnings.filterwarnings('ignore')


app = Flask(__name__)
# Corrected URI format
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'd4e4f4e9c3a0f0b7a8f9d1e2c0f7e9b1'


db = SQLAlchemy()
db.init_app(app)


class OHLC(db.Model):
    __tablename__ = 'ohlc'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    ticker_symbol = db.Column(db.String(10), nullable=False)
    open_value = db.Column(db.Float, nullable=False)
    high_value = db.Column(db.Float, nullable=False)
    low_value = db.Column(db.Float, nullable=False)
    close_value = db.Column(db.Float, nullable=False)
    pred_open_value = db.Column(db.Float, nullable=False)
    pred_high_value = db.Column(db.Float, nullable=False)
    pred_low_value = db.Column(db.Float, nullable=False)
    pred_close_value = db.Column(db.Float, nullable=False)

    def __init__(self, date, ticker_symbol, open_value, high_value, low_value, close_value, pred_open_value, pred_high_value, pred_low_value, pred_close_value):
        self.date = date
        self.ticker_symbol = ticker_symbol
        self.open_value = open_value
        self.high_value = high_value
        self.low_value = low_value
        self.close_value = close_value
        self.pred_open_value = pred_open_value
        self.pred_low_value = pred_low_value
        self.pred_high_value = pred_high_value
        self.pred_close_value = pred_close_value

def get_data(start_date, end_date, history, ticker_symbol):
    model_open = joblib.load(f'saved_models/model_open_{ticker_symbol}.joblib')
    model_low = joblib.load(f'saved_models/model_low_{ticker_symbol}.joblib')
    model_high = joblib.load(f'saved_models/model_high_{ticker_symbol}.joblib')
    model_close = joblib.load(
        f'saved_models/model_close_{ticker_symbol}.joblib')
    scaler_for_open = joblib.load(
        f'saved_models/scaler_for_open_{ticker_symbol}.joblib')
    scaler_for_high = joblib.load(
        f'saved_models/scaler_for_high_{ticker_symbol}.joblib')
    scaler_for_low = joblib.load(
        f'saved_models/scaler_for_low_{ticker_symbol}.joblib')
    scaler_for_close = joblib.load(
        f'saved_models/scaler_for_close_{ticker_symbol}.joblib')
    feature = 60
    if ticker_symbol == 'MSFT':
        feature = 2
    ticker_symbol = ticker_symbol  # Example: Apple Inc.
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='10y')
    df.reset_index(inplace=True)
    currentdate = str(datetime.now().date())
    if currentdate in str(df.iloc[-1]['Date']):
        print('date present+++++++++++++++++++++++++++++++++++')
        df = df[:-1]
    df["day"] = df["Date"].dt.day
    df["week_day"] = df["Date"].dt.weekday
    if ticker_symbol != 'MSFT':
        df["year"] = df["Date"].dt.year
        df["quarter"] = df["Date"].dt.quarter
        df["month"] = df["Date"].dt.month
    df = df.dropna()
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # 'B' is for business days
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    date_range = date_range[:10]
    timestamps = []
    for date in date_range:
        for hour in range(1):  # From 09:00 to 15:00 (inclusive)
            timestamps.append(str(date.strftime(f'%Y-%m-%d 00:00:00-04:00')))
    df_all = ticker.history(period='5y', )
    df_final = pd.DataFrame()
    df_final['Date'] = timestamps
    df_final['Open'] = np.nan
    df_final["High"] = np.nan
    df_final['Low'] = np.nan
    df_final['Close'] = np.nan
    df_final['Volume'] = np.nan
    df_final['Date'] = pd.to_datetime(df_final['Date'])
    # df["hour"] = df["Datetime"].dt.hour

    for shift in range(1, feature):
        df_final[f'Prev{shift}_Close'] = np.nan
        df_final[f'Prev{shift}_Low'] = np.nan
        df_final[f'Prev{shift}_High'] = np.nan
        df_final[f'Prev{shift}_Open'] = np.nan

    df_final['Date'] = pd.to_datetime(df_final['Date'])
    df_final["day"] = df_final["Date"].dt.day
    df_final["week_day"] = df_final["Date"].dt.weekday
    if ticker_symbol != 'MSFT':
        df_final["year"] = df_final["Date"].dt.year
        df_final["quarter"] = df_final["Date"].dt.quarter
        df_final["month"] = df_final["Date"].dt.month
    for i in range(len(df_final)):
        for shift in range(1, feature):
            df_final.at[i,
                        f'Prev{shift}_Close'] = df_all.iloc[-(shift)]['Close']
            df_final.at[i, f'Prev{shift}_Low'] = df_all.iloc[-(shift)]['Low']
            df_final.at[i, f'Prev{shift}_High'] = df_all.iloc[-(shift)]['High']
            df_final.at[i, f'Prev{shift}_Open'] = df_all.iloc[-(shift)]['Open']

        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(
            columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volume'])
        x = scaler_for_open.transform(x)
        y_pred = model_open.predict(x)
        df_final.at[i, 'Open'] = y_pred
        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(
            columns=['Date', 'Close', 'High', 'Low', 'Volume'])
        x = scaler_for_high.transform(x)
        y_pred = model_high.predict(x)
        df_final.at[i, 'High'] = y_pred
        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(
            columns=['Date', 'High', 'Close', 'Low', 'Volume'])
        x = scaler_for_low.transform(x)
        y_pred = model_low.predict(x)
        df_final.at[i, 'Low'] = y_pred
        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(columns=['Date', 'Close', 'Volume'])
        x = scaler_for_close.transform(x)
        y_pred = model_close.predict(x)
        df_final.at[i, 'Close'] = y_pred
        df_all.reset_index(inplace=True)
        df_all = pd.concat(
            [df_all, df_final.loc[[i], ['Date', 'Open', 'High', 'Low', 'Close']]])
        df_all['Date'] = pd.to_datetime(df_all['Date'], utc=True)
        df_all.set_index('Date', inplace=True)
    dfcandle = df_final[['Open', 'High', 'Low', 'Close', 'Date']]
    dfcandle_past = df[['Open', 'High', 'Low', 'Close', 'Date']]
    dfcandle_past = dfcandle_past[-history:]
    df_candlesticks = pd.concat([dfcandle_past, dfcandle])
    df_candlesticks['Date'] = pd.to_datetime(df_candlesticks['Date'], utc=True)

    return df_candlesticks

def get_stock_price(start_date, end_date, history, ticker_symbol):
    print(start_date, end_date, history, ticker_symbol, 'history========================')
    model_open = joblib.load(f'saved_models/model_open_{ticker_symbol}.joblib')
    model_low = joblib.load(f'saved_models/model_low_{ticker_symbol}.joblib')
    model_high = joblib.load(f'saved_models/model_high_{ticker_symbol}.joblib')
    model_change = joblib.load(
        f'saved_models/model_change_{ticker_symbol}.joblib')
    model_close = joblib.load(
        f'saved_models/model_close_{ticker_symbol}.joblib')
    scaler_for_open = joblib.load(
        f'saved_models/scaler_for_open_{ticker_symbol}.joblib')
    scaler_for_high = joblib.load(
        f'saved_models/scaler_for_high_{ticker_symbol}.joblib')
    scaler_for_low = joblib.load(
        f'saved_models/scaler_for_low_{ticker_symbol}.joblib')
    scaler_for_change = joblib.load(
        f'saved_models/scaler_for_change_{ticker_symbol}.joblib')
    scaler_for_close = joblib.load(
        f'saved_models/scaler_for_close_{ticker_symbol}.joblib')
    feature = 2
    ticker_symbol = ticker_symbol  # Example: Apple Inc.
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='10y')

    df.reset_index(inplace=True)
    currentdate = str(datetime.now().date())

    if currentdate in str(df.iloc[-1]['Date']):
        print('date present+++++++++++++++++++++++++++++++++++')
        df = df[:-1]
    df['Low_Open'] = (df['Low'] - df['Open']) / df['Open']
    df['Low_Open'] = df['Low_Open'].shift(1)

    df['variation'] = (df['High']-df['Low'])/df['Open']

    df['variation'] = df['variation'].shift(1)

    df['change'] = df['Close'].pct_change() * 100

    df['change'] = df['change'].shift(1)

    df['7-Day SMA'] = df['Close'].rolling(window=7).mean()
    df['7-Day SMA'] = df['7-Day SMA'].shift(1)

    # Calculate the 7-day Standard Deviation (STD)
    df['7-Day STD'] = df['Close'].rolling(window=7).std()
    df['7-Day STD'] = df['7-Day STD'].shift(1)

    # Assuming your DataFrame is named df_all
    df['High_Close'] = (df['High'] - df['Close']) / df['Open']
    df['High_Close'] = df['High_Close'].shift(1)

    df['diffrence'] = df['Open']-df['Close']

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

    df['Date'] = pd.to_datetime(df['Date'])

    df["day"] = df["Date"].dt.day
    df["week_day"] = df["Date"].dt.weekday

    df = df.dropna()
    y = df['change']

    for shifts in range(1, feature):
        df[f'Prev{shifts}_Close'] = df['Close'].shift(shifts)
        df[f'Prev{shifts}_Low'] = df['Low'].shift(shifts)
        df[f'Prev{shifts}_High'] = df['High'].shift(shifts)
        df[f'Prev{shifts}_Open'] = df['Open'].shift(shifts)

    df['Date'] = pd.to_datetime(df['Date'])
    df["day"] = df["Date"].dt.day
    df["week_day"] = df["Date"].dt.weekday
    df = df.dropna()
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # 'B' is for business days
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    date_range = date_range[:10]
    timestamps = []
    for date in date_range:
        for hour in range(1):  # From 09:00 to 15:00 (inclusive)
            timestamps.append(str(date.strftime(f'%Y-%m-%d 00:00:00-04:00')))

    print(timestamps)
    df_all = ticker.history(period='5y', )
    df_all['Low_Open'] = (df_all['Low'] - df_all['Open']) / df_all['Open']
    df_all['variation'] = (df_all['High']-df_all['Low'])/df_all['Open']
    df_all['change'] = df_all['Close'].pct_change() * 100
    df_all['7-Day SMA'] = df_all['Close'].rolling(window=7).mean()
    df_all['7-Day STD'] = df_all['Close'].rolling(window=7).std()
    df_all['High_Close'] = (df_all['High'] - df_all['Close']) / df_all['Open']
    df_all['diffrence'] = df_all['Open']-df_all['Close']
    df_final = pd.DataFrame()
    df_final['Date'] = timestamps
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
    df_final['Date'] = pd.to_datetime(df_final['Date'])
    df_final["day"] = df_final["Date"].dt.day
    df_final["week_day"] = df_final["Date"].dt.weekday
    for i in range(len(df_final)):
        df_final.at[i, 'Low_Open'] = df_all.iloc[-1]['Low_Open']
        df_final.at[i, 'variation'] = df_all.iloc[-1]['variation']
        df_final.at[i, 'change'] = df_all.iloc[-1]['change']
        df_final.at[i, '7-Day SMA'] = df_all.iloc[-1]['7-Day SMA']
        df_final.at[i, '7-Day STD'] = df_all.iloc[-1]['7-Day STD']
        df_final.at[i, 'High_Close'] = df_all.iloc[-1]['High_Close']
        df_final.at[i, 'diffrence'] = df_all.iloc[-1]['diffrence']
        for shift in range(1, feature):
            df_final.at[i,
                        f'Prev{shift}_Close'] = df_all.iloc[-(shift)]['Close']
            df_final.at[i, f'Prev{shift}_Low'] = df_all.iloc[-(shift)]['Low']
            df_final.at[i, f'Prev{shift}_High'] = df_all.iloc[-(shift)]['High']
            df_final.at[i, f'Prev{shift}_Open'] = df_all.iloc[-(shift)]['Open']
            df_final.at[i,
                        f'Prev{shift}_change'] = df_all.iloc[-(shift)]['change']

            df_final.at[i,
                        f'Prev{shift}_Low_Open'] = df_all.iloc[-(shift)]['Low_Open']
            df_final.at[i,
                        f'Prev{shift}_variation'] = df_all.iloc[-(shift)]['variation']
            df_final.at[i,
                        f'Prev{shift}_7-Day SMA'] = df_all.iloc[-(shift)]['7-Day SMA']
            df_final.at[i,
                        f'Prev{shift}_7-Day STD'] = df_all.iloc[-(shift)]['7-Day STD']
            df_final.at[i,
                        f'Prev{shift}_High_Close'] = df_all.iloc[-(shift)]['High_Close']
            df_final.at[i,
                        f'Prev{shift}_diffrence'] = df_all.iloc[-(shift)]['diffrence']

        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(
            columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volume'])
        x = scaler_for_open.transform(x)
        y_pred = model_open.predict(x)
        df_final.at[i, 'Open'] = y_pred
        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(
            columns=['Date', 'Close', 'High', 'Low', 'Volume'])
        x = scaler_for_high.transform(x)
        y_pred = model_high.predict(x)
        df_final.at[i, 'High'] = y_pred
        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(
            columns=['Date', 'High', 'Close', 'Low', 'Volume'])
        x = scaler_for_low.transform(x)
        y_pred = model_low.predict(x)
        df_final.at[i, 'Low'] = y_pred

        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(
            columns=['Date', 'Close', 'Volume', 'diffrence', 'change'])
        x = scaler_for_change.transform(x)
        y_pred = model_change.predict(x)
        df_final.at[i, 'diffrence'] = y_pred

        df_for_pred = df_final.iloc[[i], :]
        x = df_for_pred.drop(columns=['Date', 'Close', 'Volume'])
        x = scaler_for_close.transform(x)
        y_pred = model_close.predict(x)
        df_final.at[i, 'Close'] = y_pred

        df_final.at[i, 'variation'] = (
            df_final.iloc[i]['High']-df_final.iloc[i]['Low'])/df_final.iloc[i]['Open']

        df_all.reset_index(inplace=True)
        df_final.at[i, 'Low_Open'] = (
            df_final.iloc[i]['Low'] - df_final.iloc[i]['Open']) / df_final.iloc[i]['Open']
        df_final.at[i, 'High_Close'] = (
            df_final.iloc[i]['High'] - df_final.iloc[i]['Close']) / df_final.iloc[i]['Open']
        df_final.at[i, 'diffrence'] = df_final.iloc[i]['Open'] - \
            df_final.iloc[i]['Close']

        df_all = pd.concat([df_all, df_final.loc[[i], ['Date', 'Open', 'High', 'Low',
                           'Close', 'Low_Open', 'variation', 'change', 'High_Close', 'diffrence']]])
        df_all['Date'] = pd.to_datetime(df_all['Date'], utc=True)
        df_all.set_index('Date', inplace=True)
        df_all['change'] = df_all['Close'].pct_change() * 100
        df_all['7-Day SMA'] = df_all['Close'].rolling(window=7).mean()
        df_all['7-Day STD'] = df_all['Close'].rolling(window=7).std()

    print(df_final[['Open', 'High', 'Low', 'Close', 'Date']])
    dfcandle = df_final[['Open', 'High', 'Low', 'Close', 'Date']]
    dfcandle_past = df[['Open', 'High', 'Low', 'Close', 'Date']]
    dfcandle_past = dfcandle_past[-history:]
    df_candlesticks = pd.concat([dfcandle_past, dfcandle])
    df_candlesticks['Date'] = pd.to_datetime(df_candlesticks['Date'], utc=True)
    print("tdhhhhhhh",df_candlesticks)
    return df_candlesticks

@app.route('/performance', methods=['GET', 'POST'])
def performance():
    print(OHLC.query.all(), '===========')
    ticker_symbol = 'MSFT'
    date = datetime.now().date()
    date = date.strftime('%Y-%m-%d')
    print("efdefsfr", type(date), date)
    end_date = datetime.now().date()

    if request.method == 'POST':
        ticker_symbol = request.form.get('ticker_symbol')
        date = request.form.get('date')
        print(ticker_symbol, date, "ticker_symbol,date")
    previous_data = 150

    ticker = yf.Ticker(ticker_symbol)
    print("efkjmfzkmdk", type(date), date)
    df = ticker.history(start=date, end=end_date)

    print("GRESJHK", ticker, date, df)

    df.reset_index(inplace=True)
    data = df[['Open', 'High', 'Low', 'Close', 'Date']]
    # data = data[-150:]
    print("GRESJHK", data)
    # Extracting values into lists
    x = list(data['Date'])
    open_list = list(data['Open'])
    high_list = list(data['High'])
    low_list = list(data['Low'])
    close_list = list(data['Close'])
    date_list = list(data['Date'])
    print("fgggggggg", date_list, open_list, open_list, low_list, close_list)
    # Zipping the lists together for historical data
    historical = zip(date_list, open_list, high_list, low_list, close_list)

    # Fetching all assumptions from the database for a specific ticker symbol
    assumptions_records = OHLC.query.filter_by(
        ticker_symbol=ticker_symbol, date=date)

    print('All Records:', assumptions_records)
    # Initialize lists for each column
    dates = []
    assumed_opens = []
    assumed_highs = []
    assumed_lows = []
    assumed_closes = []

# Iterate through all records and append values to corresponding lists
    for record in assumptions_records:
        dates.append(record.date.strftime('%Y-%m-%d'))
        assumed_opens.append(record.open_value)
        assumed_highs.append(record.high_value)
        assumed_lows.append(record.low_value)
        assumed_closes.append(record.close_value)

# Debugging print statements to see the columnar data
    print('Dates:', dates)
    print('Assumed Opens:', assumed_opens)
    print('Assumed Highs:', assumed_highs)
    print('Assumed Lows:', assumed_lows)
    print('Assumed Closes:', assumed_closes)

# If you want to zip them together for further use
    assumptions = list(
        zip(dates, assumed_opens, assumed_highs, assumed_lows, assumed_closes))

    print('Assumptions (Zipped):', assumptions)




    predicted_records = OHLC.query.filter_by(
        ticker_symbol=ticker_symbol, date=date)

    print('All Records:', predicted_records)
    # Initialize lists for each column
    dates = []
    pred_opens = []
    pred_highs = []
    pred_lows = []
    pred_closes = []

# Iterate through all records and append values to corresponding lists
    for record in predicted_records:
        dates.append(record.date.strftime('%Y-%m-%d'))
        pred_opens.append(record.pred_open_value)
        pred_highs.append(record.pred_high_value)
        pred_lows.append(record.pred_low_value)
        pred_closes.append(record.pred_close_value)

# Debugging print statements to see the columnar data
    print('Dates:', dates)
    print('Predicted Opens:', pred_opens)
    print('Predicted Highs:', pred_highs)
    print('Predicted Lows:', pred_lows)
    print('Predicted Closes:', pred_closes)

# If you want to zip them together for further use
    predictions = list(
        zip(dates, pred_opens, pred_highs, pred_lows, pred_closes))

    print('Predictions (Zipped):', predictions)

    # Render the template with all the required variables
    print(historical, 'historical')
    return render_template('performance.html', historical=historical,  previous_data=previous_data, ticker_symbol=ticker_symbol, assumptions=assumptions, predictions=predictions)

@ app.route('/', methods=['GET', 'POST'])
def index():
    last_submit=''
    start_date = ''
    end_date = ''
    previous_data = 150
    ticker_symbol = 'MSFT'
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='1y')
    df.reset_index(inplace=True)
    data = df[['Open', 'High', 'Low', 'Close', 'Date']]
    data = data[-150:]
    historical=[]
    assumptions=[]
    predictions=[]
    ohlc_date=''
    if True:
        if request.method == 'POST':
            print(request.form,'requests')

            if 'comparison' in request.form:
                last_submit='comparison'
                start_date = request.form.get('start_date')
                end_date = request.form.get('end_date')
                ticker_symbol = request.form.get('ticker_symbol')
                date = request.form.get('date')
                print(date)
                startdate=datetime.strptime(date,'%Y-%m-%d')
                enddate=startdate+timedelta(days=1)
                previous_data = 150

                ticker = yf.Ticker(ticker_symbol)
                
                df = ticker.history(start=startdate, end=enddate)

                
                print(df,'---------------===========')
                df.reset_index(inplace=True)
                datas = df[['Open', 'High', 'Low', 'Close', 'Date']]
                # data = data[-150:]

                # Extracting values into lists
                x = list(datas['Date'])
                open_list = list(datas['Open'])
                high_list = list(datas['High'])
                low_list = list(datas['Low'])
                close_list = list(datas['Close'])
                date_list = list(datas['Date'])

                # Zipping the lists together for historical data
                historical = zip(date_list, open_list, high_list, low_list, close_list)

                # Fetching all assumptions from the database for a specific ticker symbol
                assumptions_records = OHLC.query.filter_by(
                    ticker_symbol=ticker_symbol, date=date)


                # Initialize lists for each column
                dates = []
                assumed_opens = []
                assumed_highs = []
                assumed_lows = []
                assumed_closes = []

            # Iterate through all records and append values to corresponding lists
                for record in assumptions_records:
                    dates.append(record.date.strftime('%Y-%m-%d'))
                    assumed_opens.append(record.open_value)
                    assumed_highs.append(record.high_value)
                    assumed_lows.append(record.low_value)
                    assumed_closes.append(record.close_value)


            # If you want to zip them together for further use
                assumptions = list(
                    zip(dates, assumed_opens, assumed_highs, assumed_lows, assumed_closes))



                predicted_records = OHLC.query.filter_by(
                    ticker_symbol=ticker_symbol, date=date)


                # Initialize lists for each column
                dates = []
                pred_opens = []
                pred_highs = []
                pred_lows = []
                pred_closes = []

            # Iterate through all records and append values to corresponding lists
                for record in predicted_records:
                    dates.append(record.date.strftime('%Y-%m-%d'))
                    pred_opens.append(record.pred_open_value)
                    pred_highs.append(record.pred_high_value)
                    pred_lows.append(record.pred_low_value)
                    pred_closes.append(record.pred_close_value)


            # If you want to zip them together for further use
                predictions = list(
                    zip(dates, pred_opens, pred_highs, pred_lows, pred_closes))
            elif 'prediction' in request.form:
                last_submit='prediction'
                start_date = request.form.get('start_date')
                end_date = request.form.get('end_date')
                previous_data = int(request.form.get('previous_data'))
                ticker_symbol = request.form.get('ticker_symbol')
                print("TRRhxdg",start_date,end_date,ticker_symbol)
                if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                    data = get_stock_price(start_date, end_date,
                                        previous_data, ticker_symbol)
                else:
                    data = get_data(start_date, end_date, previous_data, ticker_symbol)
            
            else :
                last_submit='assumption'
                start_date = request.form.get('start_date')
                end_date = request.form.get('end_date')
                print('--',start_date,'--',end_date,'-----tt------')
                if start_date==None or start_date=='None' or start_date.strip()=='':
                    start_date = datetime.now().date()
                    start_date=start_date.strftime('%Y-%m-%d')
                if end_date==None or end_date=='None' or end_date.strip()=='':
                    end_date = datetime.now().date()
                    end_date=end_date.strftime('%Y-%m-%d')
                print(start_date,end_date,'-----tt------')
                previous_data = int(request.form.get('previous_data'))
                ticker_symbol = request.form.get('ticker_symbol')
                print("TRRhxdg",start_date,end_date,ticker_symbol)
                if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                    data = get_stock_price(start_date, end_date,
                                        previous_data, ticker_symbol)
                else:
                    data = get_data(start_date, end_date, previous_data, ticker_symbol)

                ohlc_date = request.form.get('ohlc_date')
                print(ohlc_date,'ohlc_date')
                ticker_symbol = request.form.get('ticker_symbol')
                open_value = request.form.get('open')
                high_value = request.form.get('high')
                low_value = request.form.get('low')
                close_value = request.form.get('close')
                print("Fhcbg",ticker_symbol)
                print(ohlc_date,'ohlc_date')
                date_obj = datetime.strptime(ohlc_date, '%Y-%m-%d').date()

                # Check if a record with the same date and ticker symbol already exists
                existing_record = OHLC.query.filter_by(
                    date=date_obj, ticker_symbol=ticker_symbol).first()

                if existing_record:
                    # Update the existing record with new values
                    existing_record.open_value = float(open_value)
                    existing_record.high_value = float(high_value)
                    existing_record.low_value = float(low_value)
                    existing_record.close_value = float(close_value)

                    # Optionally, you can update prediction values if needed
                    if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                        datas = get_stock_price(ohlc_date, ohlc_date, 1, ticker_symbol)
                    else:
                        print('going to predict')
                        datas = get_data(ohlc_date, ohlc_date, 1, ticker_symbol)
                    print(datas)
                    last_row = datas.iloc[-1]
                    existing_record.pred_open_value = last_row['Open']
                    existing_record.pred_high_value = last_row['High']
                    existing_record.pred_low_value = last_row['Low']
                    existing_record.pred_close_value = last_row['Close']

                    flash('OHLC data updated successfully!', 'success')
                else:
                    # Create a new OHLC record
                    if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                        datas = get_stock_price(ohlc_date, ohlc_date, 1, ticker_symbol)
                    else:
                        print('going to predict')
                        datas = get_data(ohlc_date, ohlc_date, 1, ticker_symbol)
                    print(datas)
                    last_row = datas.iloc[-1]
                    new_ohlc = OHLC(date=date_obj,
                                    ticker_symbol=ticker_symbol,
                                    open_value=float(open_value),
                                    high_value=float(high_value),
                                    low_value=float(low_value),
                                    close_value=float(close_value),
                                    pred_open_value=last_row['Open'],
                                    pred_high_value=last_row['High'],
                                    pred_low_value=last_row['Low'],
                                    pred_close_value=last_row['Close'])
                    

                    try:
                        db.session.add(new_ohlc)
                        db.session.commit()
                        flash('OHLC data added successfully!', 'success')
                        # return redirect(url_for('add_ohlc'))
                    except Exception as e:
                        db.session.rollback()
                        flash(f'Error adding OHLC data: {e}', 'danger')
    else:
        message='Something Went wrong Please Try Again !'       
    x = list(data['Date'])
    close = list(data['Close'])
    high = list(data['High'])
    low = list(data['Low'])
    open = list(data['Open'])
   
    ohlc_data = OHLC.query.filter_by(ticker_symbol=ticker_symbol).all()
    print("--------",ohlc_data,"---------------")
    ohlc_records = [
        {
            'date': record.date,
            'open': record.open_value,
            'high': record.high_value,
            'low': record.low_value,
            'close': record.close_value
        }
        for record in ohlc_data
    ]
    print(start_date,end_date)
    return render_template('aa.html',last_submit=last_submit,ohlc_date=ohlc_date ,x=x, close=close, high=high, low=low, open=open, start_date=start_date, end_date=end_date, previous_data=previous_data, ticker_symbol=ticker_symbol, ohlc_records=ohlc_records,
                            historical=historical, assumptions=assumptions, predictions=predictions)

@app.route('/add_ohlc', methods=['GET', 'POST'])
def add_ohlc():
    if request.method == 'POST':
        print(request.form)
        date = request.form.get('ohlc_date')
        ticker_symbol = request.form.get('shared_ticker_symbol')
        open_value = request.form.get('open')
        high_value = request.form.get('high')
        low_value = request.form.get('low')
        close_value = request.form.get('close')
        print("Fhcbg",ticker_symbol)
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()

        # Check if a record with the same date and ticker symbol already exists
        existing_record = OHLC.query.filter_by(
            date=date_obj, ticker_symbol=ticker_symbol).first()

        if existing_record:
            # Update the existing record with new values
            existing_record.open_value = float(open_value)
            existing_record.high_value = float(high_value)
            existing_record.low_value = float(low_value)
            existing_record.close_value = float(close_value)

            # Optionally, you can update prediction values if needed
            if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                data = get_stock_price(date, date, 1, ticker_symbol)
            else:
                print('going to predict')
                data = get_data(date, date, 1, ticker_symbol)
            print(data)
            last_row = data.iloc[-1]
            existing_record.pred_open_value = last_row['Open']
            existing_record.pred_high_value = last_row['High']
            existing_record.pred_low_value = last_row['Low']
            existing_record.pred_close_value = last_row['Close']

            flash('OHLC data updated successfully!', 'success')
        else:
            # Create a new OHLC record
            if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                data = get_stock_price(date, date, 1, ticker_symbol)
            else:
                print('going to predict')
                data = get_data(date, date, 1, ticker_symbol)
            print(data)
            last_row = data.iloc[-1]
            new_ohlc = OHLC(date=date_obj,
                            ticker_symbol=ticker_symbol,
                            open_value=float(open_value),
                            high_value=float(high_value),
                            low_value=float(low_value),
                            close_value=float(close_value),
                            pred_open_value=last_row['Open'],
                            pred_high_value=last_row['High'],
                            pred_low_value=last_row['Low'],
                            pred_close_value=last_row['Close'])
            
            try:
                db.session.add(new_ohlc)
                db.session.commit()
                flash('OHLC data added successfully!', 'success')
                # return redirect(url_for('add_ohlc'))
            except Exception as e:
                db.session.rollback()
                flash(f'Error adding OHLC data: {e}', 'danger')
        
    return redirect(url_for('index'))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        app.run(debug=True)
