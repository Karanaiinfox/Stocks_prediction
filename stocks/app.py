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
from flask import Flask, render_template, request, redirect, flash
import alpaca_trade_api as tradeapi
from flask import Flask, request, jsonify
from alpaca.trading.client import TradingClient # type: ignore
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate



# from stock import db
warnings.filterwarnings('ignore')


app = Flask(__name__)

SEC_KEY = "pg3e1tBHvvrlGjxMk6QgiMUAzJMKuW6ybI7m3Xua"
PUB_KEY = "PK1GLKD13RBD5AQNFEKF"
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API
api = tradeapi.REST(key_id=PUB_KEY, secret_key=SEC_KEY, base_url=BASE_URL)
trading_client = TradingClient(PUB_KEY,SEC_KEY,paper=True)

# Corrected URI format
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'd4e4f4e9c3a0f0b7a8f9d1e2c0f7e9b1'
db = SQLAlchemy()
migrate = Migrate(app, db)  # Initialize Flask-Migrate



db.init_app(app)

app.config['JWT_SECRET_KEY'] = 'da7787a51c44ee84aaa1de4493cdea8ff1678ad62d61c7ee6215fc20690d6e6d'  # Secret for signing JWT
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)  # ✅ Token expires in 1 day

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    wallet = db.relationship('Wallet', backref='user', uselist=False)  # One-to-One


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Wallet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)  # One-to-One with User
    available_balance = db.Column(db.Float, default=5000.0)  # Available funds
    freezed_balance = db.Column(db.Float, default=2000.0)  # Funds on hold
    token = db.Column(db.String(100), default= 'USD')  # Unique token
    status = db.Column(db.String(20), default='active')  # active/inactive
    created_at = db.Column(db.DateTime, default=datetime.now)

class UserStock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Owner of stock
    symbol = db.Column(db.String(10), nullable=False)  # Stock symbol (e.g., AAPL)
    quantity = db.Column(db.Integer, nullable=False, default=0)  # Number of shares
    avg_price = db.Column(db.Float, nullable=False)  # Average price per share
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('stocks', lazy=True))
   

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    wallet_id = db.Column(db.Integer, db.ForeignKey('wallet.id'), nullable=False)  
    alpaca_order_id = db.Column(db.String(50), unique=True, nullable=True)  # New column for Alpaca order ID

    type = db.Column(db.String(10), nullable=False)  # 'buy' or 'sell'
    before_balance = db.Column(db.Float, nullable=False)  
    after_balance = db.Column(db.Float, nullable=False)  
    comment = db.Column(db.String(255))  
    datetime = db.Column(db.DateTime, default=datetime.utcnow)  
    status = db.Column(db.String(20), default='pending')  

    wallet = db.relationship('Wallet', backref=db.backref('transactions', lazy=True))



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
    # print(df,"historu")
    df = ticker.history(period='10y')
    df.reset_index(inplace=True)
    currentdate = str(datetime.now().date())
    # print(df,"dataa")
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
    print(start_date, end_date, history, ticker_symbol,
          'history========================')
    model_open = joblib.load(f'saved_models/model_open_{ticker_symbol}.joblib')
    print("a")
    model_low = joblib.load(f'saved_models/model_low_{ticker_symbol}.joblib')
    print("a")

    model_high = joblib.load(f'saved_models/model_high_{ticker_symbol}.joblib')
    print("a")

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
    print("b")
    ticker_symbol = ticker_symbol  # Example: Apple Inc.
    print( ticker_symbol,"b")

    ticker = yf.Ticker(ticker_symbol)
    print(yf,"yffff")
    print( ticker,"b")

    df = ticker.history(period='10y')
    # df = ticker.history

    print(df,"dfdfdfdsff")

    df.reset_index(inplace=True)
    currentdate = str(datetime.now().date())

    if currentdate in str(df.iloc[-1]['Date']):
        print(df,'date present+++++++++++++++++++++++++++++++++++')
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
    print("tdhhhhhhh", df_candlesticks)
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
@login_required
def index():
    last_submit = ''
    start_date = ''
    end_date = ''
    no_days = 150
    previous_data = 150
    ticker_symbol = 'MSFT'
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='1y')
    df.reset_index(inplace=True)
    data = df[['Open', 'High', 'Low', 'Close', 'Date']]
    data = data[-150:]

    historical = []
    assumptions = []
    predictions = []
    ohlc_date = ''
    no_days = ''
    comparison_date = ''
    try:
        if request.method == 'POST':
            print(request.form, 'requests')

            if 'comparison' in request.form:
                last_submit = 'comparison'
                start_date = request.form.get('start_date')
                end_date = request.form.get('end_date')
                no_days = request.form.get('no_days', 150)
                ticker_symbol = request.form.get('ticker_symbol')
                if no_days == '':
                    no_days = 150
                else:
                    no_days = int(no_days)

                print("No_Of_Days", no_days)
                print("Type of No_Of_Days:", type(no_days))

                date = request.form.get('date')
                comparison_date = date
                print(date, "asdfghjk")
                enddate = datetime.today().date()
                enddate = enddate + timedelta(days=1)
                print(end_date, "a")

                # end_date=datetime.today().date()
                startdate = enddate - timedelta(days=no_days)
                print(startdate, "start_date")

                print(f"Fetching data from {startdate} to {enddate}")

                previous_data = 150

                ticker = yf.Ticker(ticker_symbol)

                df = ticker.history(start=startdate, end=enddate)

                print(df, '---------------===========')
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
                historical = zip(date_list, open_list,
                                 high_list, low_list, close_list)

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
                last_submit = 'prediction'
                start_date = request.form.get('start_date')
                end_date = request.form.get('end_date')
                previous_data = int(request.form.get('previous_data'))
                ticker_symbol = request.form.get('ticker_symbol')

                print("TRRhxdg", start_date, end_date, ticker_symbol)
                if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                    data = get_stock_price(start_date, end_date,
                                           previous_data, ticker_symbol)
                else:
                    data = get_data(start_date, end_date,
                                    previous_data, ticker_symbol)
                    print("TRRhxdg", start_date, end_date, ticker_symbol)

            else:
                last_submit = 'assumption'
                start_date = request.form.get('start_date')
                end_date = request.form.get('end_date')
                print('--', start_date, '--', end_date, '-----tt------')
                if start_date == None or start_date == 'None' or start_date.strip() == '':
                    start_date = datetime.now().date()
                    start_date = start_date.strftime('%Y-%m-%d')
                if end_date == None or end_date == 'None' or end_date.strip() == '':
                    end_date = datetime.now().date()
                    end_date = end_date.strftime('%Y-%m-%d')
                print(start_date, end_date, '-----tt------')
                previous_data = int(request.form.get('previous_data'))
                ticker_symbol = request.form.get('ticker_symbol')
                print("TRRhxdg", start_date, end_date, ticker_symbol)
                if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:

                    data = get_stock_price(start_date, end_date,
                                           previous_data, ticker_symbol)
                else:
                    data = get_data(start_date, end_date,
                                    previous_data, ticker_symbol)

                ohlc_date = request.form.get('ohlc_date')
                print(ohlc_date, 'ohlc_date')
                ticker_symbol = request.form.get('ticker_symbol')
                open_value = request.form.get('open')
                high_value = request.form.get('high')
                low_value = request.form.get('low')
                close_value = request.form.get('close')
                print("Fhcbg", ticker_symbol)
                print(ohlc_date, 'ohlc_date')
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
                        datas = get_stock_price(
                            ohlc_date, ohlc_date, 1, ticker_symbol)
                    else:
                        print('going to predict')
                        datas = get_data(ohlc_date, ohlc_date,
                                         1, ticker_symbol)
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
                        datas = get_stock_price(
                            ohlc_date, ohlc_date, 1, ticker_symbol)
                    else:
                        print('going to predict')
                        datas = get_data(ohlc_date, ohlc_date,
                                         1, ticker_symbol)
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

                    except Exception as e:
                        db.session.rollback()
                        flash(f'Error adding OHLC data: {e}', 'danger')
    except:
        message = 'Something Went wrong Please Try Again !'
    x = list(data['Date'])
    close = list(data['Close'])
    high = list(data['High'])
    low = list(data['Low'])
    open = list(data['Open'])

    ohlc_data = OHLC.query.filter_by(ticker_symbol=ticker_symbol).all()
    print("--------", ohlc_data, "---------------")
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
    print(start_date, end_date)
    return render_template('option01.html', comparison_date=comparison_date, no_days=no_days, last_submit=last_submit, ohlc_date=ohlc_date, x=x, close=close, high=high, low=low, open=open, start_date=start_date, end_date=end_date, previous_data=previous_data, ticker_symbol=ticker_symbol, ohlc_records=ohlc_records,
                           historical=historical, assumptions=assumptions, predictions=predictions)

@app.route('/option02', methods=['GET', 'POST'])
@login_required
def option02():
    # Initialize variables with default values
    last_submit = ''
    start_date = ''
    end_date = ''
    no_days = 150
    previous_data = 150
    ticker_symbol = 'MSFT'
    ohlc_date = ''  # Initialize ohlc_date
    comparison_date = ''
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='1y')
    df.reset_index(inplace=True)
    data = df[['Open', 'High', 'Low', 'Close', 'Date']]
    data = data[-150:]
    historical = []
    assumptions = []
    predictions = []
    historical_records = []
    dfsingles = []

    if request.method == 'POST':
        print(request.form, 'requests')

        if 'comparison' in request.form:
            last_submit = 'comparison'
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            no_days = int(request.form.get('no_days', 150))
            ticker_symbol = request.form.get('ticker_symbol')
            comparison_date = request.form.get('date')
            comparison_date = datetime.strptime(
                comparison_date, '%Y-%m-%d').date()

            historical_df = df
            historical_df['Date'] = pd.to_datetime(
                historical_df['Date']).dt.date
            filtered_data = historical_df[historical_df['Date'] == comparison_date][[
                'Date', 'Open', 'High', 'Low', 'Close']]
            dfsingle = pd.DataFrame(filtered_data)
            dates = list(filtered_data['Date'])
            opens = list(filtered_data['Open'])
            highs = list(filtered_data['High'])
            lows = list(filtered_data['Low'])
            closes = list(filtered_data['Close'])
            historical_records = list(zip(dates, opens, highs, lows, closes))

            enddate = datetime.today().date() + timedelta(days=1)
            startdate = enddate - timedelta(days=no_days)
            print(f"Fetching data from {startdate} to {enddate}")

            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=startdate, end=enddate)
            df.reset_index(inplace=True)
            datas = df[['Open', 'High', 'Low', 'Close', 'Date']]
            date_list = list(datas['Date'])
            open_list = list(datas['Open'])
            high_list = list(datas['High'])
            low_list = list(datas['Low'])
            close_list = list(datas['Close'])
            historical = list(
                zip(date_list, open_list, high_list, low_list, close_list))

            assumptions_records = OHLC.query.filter_by(
                ticker_symbol=ticker_symbol, date=comparison_date)
            dates = []
            assumed_opens = []
            assumed_highs = []
            assumed_lows = []
            assumed_closes = []
            for record in assumptions_records:
                dates.append(record.date.strftime('%Y-%m-%d'))
                assumed_opens.append(record.open_value)
                assumed_highs.append(record.high_value)
                assumed_lows.append(record.low_value)
                assumed_closes.append(record.close_value)
            assumptions_df = pd.DataFrame({
                'Date': dates,
                'Open': assumed_opens,
                'High': assumed_highs,
                'Low': assumed_lows,
                'Close': assumed_closes
            })
            dfsingle = pd.concat([dfsingle, assumptions_df], ignore_index=True)
            assumptions = list(
                zip(dates, assumed_opens, assumed_highs, assumed_lows, assumed_closes))

            predicted_records = OHLC.query.filter_by(
                ticker_symbol=ticker_symbol, date=comparison_date)
            dates = []
            pred_opens = []
            pred_highs = []
            pred_lows = []
            pred_closes = []
            for record in predicted_records:
                dates.append(record.date.strftime('%Y-%m-%d'))
                pred_opens.append(record.pred_open_value)
                pred_highs.append(record.pred_high_value)
                pred_lows.append(record.pred_low_value)
                pred_closes.append(record.pred_close_value)
            predictions_df = pd.DataFrame({
                'Date': dates,
                'Open': pred_opens,
                'High': pred_highs,
                'Low': pred_lows,
                'Close': pred_closes
            })
            dfsingle = pd.concat([dfsingle, predictions_df], ignore_index=True)
            dfsingles = list(zip(dfsingle['Date'], dfsingle['Open'],
                             dfsingle['High'], dfsingle['Low'], dfsingle['Close']))
            predictions = list(
                zip(dates, pred_opens, pred_highs, pred_lows, pred_closes))

        elif 'prediction' in request.form:
            last_submit = 'prediction'
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            previous_data = int(request.form.get('previous_data'))
            ticker_symbol = request.form.get('ticker_symbol')
            if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                data = get_stock_price(
                    start_date, end_date, previous_data, ticker_symbol)
            else:
                data = get_data(start_date, end_date,
                                previous_data, ticker_symbol)

        else:
            ohlc_date = request.form.get(
                'ohlc_date', '')  # Initialize ohlc_date
            ticker_symbol = request.form.get('ticker_symbol')
            open_value = request.form.get('open')
            high_value = request.form.get('high')
            low_value = request.form.get('low')
            close_value = request.form.get('close')
            date_obj = datetime.strptime(ohlc_date, '%Y-%m-%d').date()
            existing_record = OHLC.query.filter_by(
                date=date_obj, ticker_symbol=ticker_symbol).first()
            if existing_record:
                existing_record.open_value = float(open_value)
                existing_record.high_value = float(high_value)
                existing_record.low_value = float(low_value)
                existing_record.close_value = float(close_value)

                if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                    datas = get_stock_price(
                        ohlc_date, ohlc_date, 1, ticker_symbol)
                    print("ASDFG", ticker_symbol)
                    print("Dataaaaa", datas)
                else:
                    datas = get_data(ohlc_date, ohlc_date, 1, ticker_symbol)
                print("qwertyui", ticker_symbol)
                print(datas)
                last_row = datas.iloc[-1]
                existing_record.pred_open_value = last_row['Open']
                existing_record.pred_high_value = last_row['High']
                existing_record.pred_low_value = last_row['Low']
                existing_record.pred_close_value = last_row['Close']
                flash('OHLC data updated successfully!', 'success')

            else:
                if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                    datas = get_stock_price(
                        ohlc_date, ohlc_date, 1, ticker_symbol)
                    print("ticker", ticker_symbol)
                    print(datas, "data")
                else:
                    datas = get_data(ohlc_date, ohlc_date, 1, ticker_symbol)
                print("ticker1", ticker_symbol)
                print("data", datas)
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
                                pred_close_value=last_row['Close']
                                )
                try:
                    db.session.add(new_ohlc)
                    db.session.commit()
                    flash('OHLC data added successfully!', 'success')
                except Exception as e:
                    db.session.rollback()
                    flash(f'Error adding OHLC data: {e}', 'danger')

    x = list(data['Date'])
    close = list(data['Close'])
    high = list(data['High'])
    low = list(data['Low'])
    open = list(data['Open'])
    print("recordsList")

    ohlc_data = OHLC.query.filter_by(ticker_symbol=ticker_symbol).all()
    print("--------", ohlc_data, "---------------")
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

    return render_template('option02.html', comparison_date=comparison_date, no_days=no_days, last_submit=last_submit, ohlc_date=ohlc_date, x=x, close=close, high=high, low=low, open=open, previous_data=previous_data, ticker_symbol=ticker_symbol,
                           ohlc_records=ohlc_records, historical=historical, assumptions=assumptions, predictions=predictions, historical_records=historical_records, dfsingles=dfsingles
                           )



@app.route('/tradingview', methods=['GET', 'POST'])
@login_required
def tradingview():
    # Initialize variables with default values
    last_submit = ''
    start_date = ''
    end_date = ''
    no_days = 150
    previous_data = 150
    ticker_symbol = 'MSFT'
    ohlc_date = ''  # Initialize ohlc_date
    comparison_date = ''
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='1y')
    df.reset_index(inplace=True)
    data = df[['Open', 'High', 'Low', 'Close', 'Date']]
    data = data[-150:]
    historical = []
    assumptions = []
    predictions = []
    historical_records = []
    dfsingles = []

    if request.method == 'POST':
        print(request.form, 'requests')

        if 'comparison' in request.form:
            last_submit = 'comparison'
            start_date = request.form.get('start_date')
            print(start_date,"start date")
            end_date = request.form.get('end_date')
            print(end_date,"end_date")

            no_days = int(request.form.get('no_days', 150))
            ticker_symbol = request.form.get('ticker_symbol')
            comparison_date = request.form.get('date')
            comparison_date = datetime.strptime(
                comparison_date, '%Y-%m-%d').date()

            historical_df = df
            historical_df['Date'] = pd.to_datetime(
                historical_df['Date']).dt.date
            filtered_data = historical_df[historical_df['Date'] == comparison_date][[
                'Date', 'Open', 'High', 'Low', 'Close']]
            dfsingle = pd.DataFrame(filtered_data)
            dates = list(filtered_data['Date'])
            opens = list(filtered_data['Open'])
            highs = list(filtered_data['High'])
            lows = list(filtered_data['Low'])
            closes = list(filtered_data['Close'])
            historical_records = list(zip(dates, opens, highs, lows, closes))

            enddate = datetime.today().date() + timedelta(days=1)
            startdate = enddate - timedelta(days=no_days)
            print(f"Fetching data from {startdate} to {enddate}")

            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=startdate, end=enddate)
            df.reset_index(inplace=True)
            datas = df[['Open', 'High', 'Low', 'Close', 'Date']]
            date_list = list(datas['Date'])
            open_list = list(datas['Open'])
            high_list = list(datas['High'])
            low_list = list(datas['Low'])
            close_list = list(datas['Close'])
            historical = list(
                zip(date_list, open_list, high_list, low_list, close_list))

            assumptions_records = OHLC.query.filter_by(
                ticker_symbol=ticker_symbol, date=comparison_date)
            dates = []
            assumed_opens = []
            assumed_highs = []
            assumed_lows = []
            assumed_closes = []
            for record in assumptions_records:
                dates.append(record.date.strftime('%Y-%m-%d'))
                assumed_opens.append(record.open_value)
                assumed_highs.append(record.high_value)
                assumed_lows.append(record.low_value)
                assumed_closes.append(record.close_value)
            assumptions_df = pd.DataFrame({
                'Date': dates,
                'Open': assumed_opens,
                'High': assumed_highs,
                'Low': assumed_lows,
                'Close': assumed_closes
            })
            dfsingle = pd.concat([dfsingle, assumptions_df], ignore_index=True)
            assumptions = list(
                zip(dates, assumed_opens, assumed_highs, assumed_lows, assumed_closes))

            predicted_records = OHLC.query.filter_by(
                ticker_symbol=ticker_symbol, date=comparison_date)
            dates = []
            pred_opens = []
            pred_highs = []
            pred_lows = []
            pred_closes = []
            for record in predicted_records:
                dates.append(record.date.strftime('%Y-%m-%d'))
                pred_opens.append(record.pred_open_value)
                pred_highs.append(record.pred_high_value)
                pred_lows.append(record.pred_low_value)
                pred_closes.append(record.pred_close_value)
            predictions_df = pd.DataFrame({
                'Date': dates,
                'Open': pred_opens,
                'High': pred_highs,
                'Low': pred_lows,
                'Close': pred_closes
            })
            dfsingle = pd.concat([dfsingle, predictions_df], ignore_index=True)
            dfsingles = list(zip(dfsingle['Date'], dfsingle['Open'],
                             dfsingle['High'], dfsingle['Low'], dfsingle['Close']))
            predictions = list(
                zip(dates, pred_opens, pred_highs, pred_lows, pred_closes))

        elif 'prediction' in request.form:
            last_submit = 'prediction'
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            previous_data = int(request.form.get('previous_data'))
            ticker_symbol = request.form.get('ticker_symbol')
            if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                data = get_stock_price(
                    start_date, end_date, previous_data, ticker_symbol)
            else:
                data = get_data(start_date, end_date,
                                previous_data, ticker_symbol)

        else:
            ohlc_date = request.form.get(
                'ohlc_date', '')  # Initialize ohlc_date
            ticker_symbol = request.form.get('ticker_symbol')
            open_value = request.form.get('open')
            print(open_value,"open value")
            high_value = request.form.get('high')
            print(high_value,"high value")

            low_value = request.form.get('low')
            close_value = request.form.get('close')
            date_obj = datetime.strptime(ohlc_date, '%Y-%m-%d').date()
            print(date_obj,"date onject")
            existing_record = OHLC.query.filter_by(
                date=date_obj, ticker_symbol=ticker_symbol).first()
            
            if existing_record:
                existing_record.open_value = float(open_value)
                existing_record.high_value = float(high_value)
                existing_record.low_value = float(low_value)
                existing_record.close_value = float(close_value)

                if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                    datas = get_stock_price(
                        ohlc_date, ohlc_date, 1, ticker_symbol)
                    print("ASDFG", ticker_symbol)
                    print("Dataaaaa", datas)
                else:
                    datas = get_data(ohlc_date, ohlc_date, 1, ticker_symbol)
                print("qwertyui", ticker_symbol)
                print(datas)
                last_row = datas.iloc[-1]
                existing_record.pred_open_value = last_row['Open']
                existing_record.pred_high_value = last_row['High']
                existing_record.pred_low_value = last_row['Low']
                existing_record.pred_close_value = last_row['Close']
                flash('OHLC data updated successfully!', 'success')

            else:
                if 'MSFT' in ticker_symbol or 'NFLX' in ticker_symbol:
                    datas = get_stock_price(
                        ohlc_date, ohlc_date, 1, ticker_symbol)
                    print("ticker", ticker_symbol)
                    print(datas, "data")
                else:
                    datas = get_data(ohlc_date, ohlc_date, 1, ticker_symbol)
                print("ticker1", ticker_symbol)
                print("data", datas)
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
                                pred_close_value=last_row['Close']
                                )
                try:
                    db.session.add(new_ohlc)
                    db.session.commit()
                    flash('OHLC data added successfully!', 'success')
                except Exception as e:
                    db.session.rollback()
                    flash(f'Error adding OHLC data: {e}', 'danger')

    x = list(data['Date'])
    close = list(data['Close'])
    high = list(data['High'])
    low = list(data['Low'])
    open = list(data['Open'])
    print("recordsList")

    ohlc_data = OHLC.query.filter_by(ticker_symbol=ticker_symbol).all()
    print("--------", ohlc_data, "---------------")
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

    return render_template('tradingview.html', comparison_date=comparison_date, no_days=no_days, last_submit=last_submit, ohlc_date=ohlc_date, x=x, close=close, high=high, low=low, open=open, previous_data=previous_data, ticker_symbol=ticker_symbol,
                           ohlc_records=ohlc_records, historical=historical, assumptions=assumptions, predictions=predictions, historical_records=historical_records, dfsingles=dfsingles
                           )


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
        print("Fhcbg", ticker_symbol)
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

            except Exception as e:
                db.session.rollback()
                flash(f'Error adding OHLC data: {e}', 'danger')

    return redirect(url_for('index'))

# import requests

# def get_alpaca_stock_price(symbol):
#     API_KEY = "PK1GLKD13RBD5AQNFEKF"
#     API_SECRET = "pg3e1tBHvvrlGjxMk6QgiMUAzJMKuW6ybI7m3Xua"
#     BASE_URL = "https://data.alpaca.markets/v2"

#     try:
#         headers = {
#             "APCA-API-KEY-ID": API_KEY,
#             "APCA-API-SECRET-KEY": API_SECRET
#         }
#         response = requests.get(f"{BASE_URL}/stocks/{symbol}/quotes/latest", headers=headers)
#         response.raise_for_status()  # Raise exception for bad status codes
#         quote = response.json()["quote"]

#         return {
#             "symbol": symbol,
#             "ask_price": float(quote["ap"]),  # Ask price (to buy)
#             "bid_price": float(quote["bp"]),  # Bid price (to sell)
#             "timestamp": quote["t"]  # ISO timestamp
#         }
#     except requests.exceptions.RequestException as e:
#         return {"error": f"Failed to fetch quote: {str(e)}"}
#     except Exception as e:
#         return {"error": str(e)}

@app.route('/trade')
@login_required
def index1():
    return render_template('trade.html')



# @app.route('/buy', methods=['POST'])
# @login_required
# def buy_stock():
#     try:
#         data = request.get_json()  # Get JSON data from fetch request
#         print(data,"Data ")
#         symbol = data.get('symbol')
#         qty = int(data.get('quantity', 1))  # Default: 1 share
#         price = int(data.get('price'))  # Default: 1 share
 

        
#         print(f"Buying {qty} shares of {symbol}")  # Debugging print


#         # Submit buy order
#         api.submit_order(
#             symbol=symbol,
#             qty=qty,
#             side='buy',
#             type='market',
#             time_in_force='gtc'
        
#         )

#         return jsonify({
#             "message": f"Successfully bought {qty} shares of {symbol} at ${price}!",
#             "price": price
#         }), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400
from flask_jwt_extended import decode_token

@app.route('/buy', methods=['POST'])
@login_required
def buy_stock():
    if True:
        data = request.get_json()
        print(data, "Received Data")

        symbol = data.get('symbol')
        qty = int(data.get('quantity', 1))
        price = data.get('price')
        print(price, "Received Price")

        # Find the user's wallet
        token = session.get("access_token")
        print(token, "Token Received")

        decoded_token = decode_token(token)
        user_id = decoded_token.get("sub") 
        print("Decoded User ID:", user_id)

        wallet = Wallet.query.filter_by(user_id=user_id).first()
        if not wallet:
            return jsonify({"error": "Wallet not found"}), 400

        price = float(price)
        total_cost = qty * price  
        print(total_cost, "Total Cost")

        # Check if the user has enough balance
        if wallet.available_balance < total_cost:
            return jsonify({"error": "Insufficient balance"}), 400

        # Attempt to submit the buy order to the API first
        try:
            order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
            order_id = order.id 
            print(order_id,"oddr id is ")
        except Exception as e:
            error_message = str(e)
            print("API Error:", error_message)

            # Specific error handling if API submission fails
            if "insufficient buying power" in error_message.lower():
                return jsonify({"error": "Insufficient buying power. Please check your balance."}), 400
            
            return jsonify({"error": "Order could not be placed. Please try again later."}), 400

        # If API order is successfully placed, update the database
        before_balance = wallet.available_balance
        after_balance = before_balance - total_cost
        print(before_balance,"oooo")
        # Create a transaction record
        transaction = Transaction(
            wallet_id=wallet.id,
            type='buy',
            before_balance=before_balance,
            after_balance=after_balance,
            comment=f"Bought {qty} shares of {symbol} at ${price}",
            status="success",
            alpaca_order_id=order_id  # Store order ID

        )
        print("bchagghcgsddddddddddddddc")

        # Deduct the balance
        wallet.available_balance = after_balance

        # Check if user already owns this stock
        user_stock = UserStock.query.filter_by(user_id=user_id, symbol=symbol).first()
        if user_stock:
            # Update existing stock quantity and average price
            new_total_shares = user_stock.quantity + qty
            new_avg_price = ((user_stock.quantity * user_stock.avg_price) + (qty * price)) / new_total_shares
            user_stock.quantity = new_total_shares
            user_stock.avg_price = new_avg_price
        else:
            # Create new stock entry
            user_stock = UserStock(user_id=user_id, symbol=symbol, quantity=qty, avg_price=price)
            db.session.add(user_stock)

        # Commit changes to the database only after successful order placement
        print("befo")
        db.session.add(transaction)
        print("af")

        db.session.commit()
        print("af..")


        return jsonify({
            "message": f"Successfully bought {qty} shares of {symbol} at ${price}!",
            "price": price,
            "order_id": order_id
        }), 200

    # except Exception as e:
    #     return jsonify({"error": "An unexpected error occurred. Please try again later."}), 400




# @app.route('/orders', methods=['GET'])
# @login_required
# def get_user_orders():
#     try:
#         # Get user ID
#         token = session.get("access_token")
#         decoded_token = decode_token(token)
#         user_id = decoded_token.get("sub")  

#         # Fetch all orders from Alpaca
#         try:
#             alpaca_orders = api.list_orders(status="all", limit=50)  # Fetch all recent orders
#         except Exception as e:
#             return jsonify({"error": f"Failed to fetch orders: {str(e)}"}), 400

#         # Filter orders for this user
#         user_transactions = Transaction.query.filter_by(wallet_id=Wallet.id, type='buy').all()
#         user_order_ids = {t.alpaca_order_id for t in user_transactions}  # Get stored order IDs

#         user_orders = []
#         for order in alpaca_orders:
#             if order.id in user_order_ids:
#                 user_orders.append({
#                     "order_id": order.id,
#                     "symbol": order.symbol,
#                     "qty": order.qty,
#                     "filled_qty": order.filled_qty,
#                     "price": order.limit_price or "Market",
#                     "status": order.status,
#                     "side": order.side,
#                     "submitted_at": order.submitted_at
#                 })

#         return jsonify({"orders": user_orders}), 200

#     except Exception as e:
#         return jsonify({"error": "An unexpected error occurred while fetching orders."}), 400


@app.route('/orders', methods=['GET'])
@login_required
def get_user_orders():
    try:
        # Get user ID from session token
        token = session.get("access_token")
        decoded_token = decode_token(token)
        user_id = decoded_token.get("sub")  

        # Fetch the wallet associated with this user
        user_wallet = Wallet.query.filter_by(user_id=user_id).first()
        if not user_wallet:
            return jsonify({"error": "Wallet not found for the user"}), 404

        # Fetch all orders from Alpaca
        try:
            alpaca_orders = api.list_orders(status="all", limit=50)  # Fetch all recent orders
        except Exception as e:
            return jsonify({"error": f"Failed to fetch orders: {str(e)}"}), 400

        # Fetch user's transactions only (filter by wallet ID)
        user_transactions = Transaction.query.filter_by(wallet_id=user_wallet.id, type='buy').all()
        user_order_ids = {t.alpaca_order_id for t in user_transactions}  # Get stored order IDs

        user_orders = []
        for order in alpaca_orders:
            if order.id in user_order_ids:
                user_orders.append({
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "qty": order.qty,
                    "filled_qty": order.filled_qty,
                    "price": order.limit_price or "Market",
                    "status": order.status,
                    "side": order.side,
                    "submitted_at": order.submitted_at
                })

        return jsonify({"orders": user_orders}), 200

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred while fetching orders.", "details": str(e)}), 400


@app.route('/sell', methods=['POST'])
@login_required
def sell_stock():
    try:
        data = request.get_json()
        print(data, "Received Data")

        symbol = data.get('symbol')
        qty = int(data.get('quantity', 1))
        price = data.get('price')

        print(price, "Received Price")

        # Get user ID from session token
        token = session.get("access_token")
        print(token, "Token Received")

        decoded_token = decode_token(token)
        user_id = decoded_token.get("sub") 
        print("Decoded User ID:", user_id)

        # Fetch user wallet
        wallet = Wallet.query.filter_by(user_id=user_id).first()
        if not wallet:
            return jsonify({"error": "Wallet not found"}), 400

        # Ensure price is a valid float
        price = float(price)

        # Check if user owns enough stock to sell
        user_stock = UserStock.query.filter_by(user_id=user_id, symbol=symbol).first()
        if not user_stock or user_stock.quantity < qty:
            return jsonify({"error": "Insufficient stock quantity"}), 400

        # Attempt to submit the sell order
        try:
            order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
            order_id = order.id 
            print(order_id,"ordererere")

        except Exception as e:
            error_message = str(e)
            print("API Error:", error_message)
            
            # Specific error handling for wash trade
            if "potential wash trade detected" in error_message.lower():
                return jsonify({"error": "Potential wash trade detected. Use complex orders."}), 400
            
            # General error message for any other API failure
            return jsonify({"error": "Order could not be placed. Please try again later."}), 400

        # If API order is successfully placed, proceed with database update
        total_sell_value = qty * price  
        before_balance = wallet.available_balance
        after_balance = before_balance + total_sell_value  # Add sold amount to wallet

        # Create a transaction record
        transaction = Transaction(
            wallet_id=wallet.id,
            type='sell',
            before_balance=before_balance,
            after_balance=after_balance,
            comment=f"Sold {qty} shares of {symbol} at ${price}",
            status="success",
            alpaca_order_id=order_id  # Store order ID

        )
        print("jgshsadhfadsfasd")
        # Update wallet balance
        wallet.available_balance = after_balance
        print("jg...............shsadhfadsfasd")


        # Update or delete stock record
        if user_stock.quantity == qty:
            db.session.delete(user_stock)  # If selling all, remove stock entry
        else:
            user_stock.quantity -= qty  # Reduce quantity

        # Commit changes
        db.session.add(transaction)
        db.session.commit()

        return jsonify({
            "message": f"Successfully sold {qty} shares of {symbol} at ${price}!",
            "price": price,
            "order_id": order_id

        }), 200

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 400




@app.route('/get_orders', methods=['GET'])
@login_required
def get_orders():
    try:
        # Optional: Filter orders (e.g., only open or all)
        request_params = GetOrdersRequest(
            status=QueryOrderStatus.ALL,  # or OPEN, CLOSED
            limit=50  # Max 500 per request
        )
        orders = trading_client.get_orders(filter=request_params)
        print(orders,"orders are")

        # Format the orders for response
        orders_list = [
            {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "status": order.status,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "created_at": order.created_at.isoformat()
            }
            for order in orders
        ]

        return jsonify({
            "message": "Orders retrieved successfully",
            "orders": orders_list
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        new_wallet = Wallet(user_id=new_user.id)
        db.session.add(new_wallet)
        db.session.commit()

        flash('Registration successful. You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            #session,token(jwt)
            # whenver user login then new token generate
            # time limit : 1 day
            # Ensure the user has a wallet; create if missing
            access_token = create_access_token(identity=str(user.id), expires_delta=timedelta(days=1))
            session['access_token']=access_token
            
            print(access_token,"acces token are")

            if not Wallet.query.filter_by(user_id=user.id).first():
                new_wallet = Wallet(user_id=user.id)
                db.session.add(new_wallet)
                db.session.commit()

            flash('Login successful!', 'success')
            return redirect(url_for('index'))  # Redirect to the main page
            # return jsonify({
            #     "message": "Login successful!",
            #     "access_token": access_token
            # }), 200

        flash('Invalid credentials', 'danger')

    return render_template('login.html')


from flask import session





@app.route('/logout')
@login_required  # Ensures only logged-in users can access this
def logout():
    session.clear()
    flash('You have been logged out.', 'success')

    return redirect(url_for('login'))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        app.run(debug=True)
