# # from app import app, db, Wallet

# # def delete_wallet(user_id):
# #     with app.app_context():  # ✅ Ensure the app context is active
# #         wallet = Wallet.query.filter_by(user_id=user_id).first()
# #         if wallet:
# #             db.session.delete(wallet)
# #             db.session.commit()
# #             print(f"Wallet for user {user_id} deleted successfully.")
# #         else:
# #             print(f"No wallet found for user {user_id}.")

# # # Example: Delete wallet for user ID 3
# # delete_wallet(3)
# # from app import db, app
# # from sqlalchemy import text  # ✅ Import text()

# # def drop_wallet_table():
# #     with app.app_context():  # Ensure app context is active
# #         db.session.execute(text("DROP TABLE IF EXISTS wallet"))  # ✅ Wrap in text()
# #         db.session.commit()
# #         print("Wallet table deleted successfully.")

# # # Run the function
# # drop_wallet_table()


# # import secrets
# # print(secrets.token_hex(32))

# import alpaca_trade_api as tradeapi
# from app import app
# # SEC_KEY = "pg3e1tBHvvrlGjxMk6QgiMUAzJMKuW6ybI7m3Xua"
# # PUB_KEY = "PK1GLKD13RBD5AQNFEKF"
# # BASE_URL = "https://paper-api.alpaca.markets"
# # alpaca = tradeapi.REST(SEC_KEY, PUB_KEY, BASE_URL)
# # @app.route('/stock-get', methods=['GET', 'POST'])
# # def get_stock_price(symbol='TSLA'):
# #     try:
# #         barset = alpaca.get_latest_trade(symbol)

# #         print(barset.price)
# #         return barset
# #     except Exception as e:
# #         print("Error fetching stock price:", e)
# #         return None



# # from flask import Flask, jsonify, request
# # from alpaca_trade_api.rest import REST, TimeFrame
# # from alpaca.data.requests import StockLatestQuoteRequest
# # from alpaca.data import StockDataClient  # Correct import
# # app = Flask(__name__)

# # Replace with your Alpaca API keys
# # API_KEY = "PK1GLKD13RBD5AQNFEKF"
# # SECRET_KEY = "pg3e1tBHvvrlGjxMk6QgiMUAzJMKuW6ybI7m3Xua"

# # Initialize Alpaca API client
# # alpaca = REST(API_KEY, SECRET_KEY, base_url="https://paper-api.alpaca.markets")

# # Route to fetch latest stock price
# # @app.route('/get_stock_prices', methods=['GET'])
# # def get_stock_price():
# #     symbol = request.args.get('symbol')
# #     if not symbol:
# #         return jsonify({"error": "Missing symbol"}), 400

# #     try:
# #         request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
# #         latest_quote = data_client.get_stock_latest_quote(request_params)
# #         quote = latest_quote[symbol]  # Get quote for the specific symbol

# #         return jsonify({
# #             "symbol": symbol,
# #             "ask_price": float(quote.ask_price),  # Price to buy
# #             "bid_price": float(quote.bid_price),  # Price to sell
# #             "timestamp": quote.timestamp.isoformat()
# #         }), 200

# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500


    
# # if __name__ == "__main__":
# #     with app.app_context():
# #         app.run(debug=True)


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

# # Example usage in Flask route
# from flask import Flask, jsonify

# app = Flask(__name__)

# @app.route('/get_stock_prices', methods=['GET'])
# def get_stock_price():
#     symbol = request.args.get('symbol')
#     if not symbol:
#         return jsonify({"error": "Missing symbol"}), 400
#     result = get_alpaca_stock_price(symbol)
#     if "error" in result:
#         return jsonify(result), 500
#     return jsonify(result), 200


from app import UserStock
from app import app


with app.app_context():

  

    stocks = UserStock.query.all()
    for stock in stocks:
        print(stock.user_id, stock.symbol, stock.quantity)
        print("dsd")

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
        print(user_stock,"usersotvk")
        print(user_stock.quantity,"qunatitgy")
        if not user_stock or user_stock.quantity < qty:
            return jsonify({"error": "Insufficient stock quantity"}), 400
        
        # Calculate total sell value
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
            status="success"
        )

        # Update wallet balance
        wallet.available_balance = after_balance

        # Update or delete stock record
        if user_stock.quantity == qty:
            db.session.delete(user_stock)  # If selling all, remove stock entry
        else:
            user_stock.quantity -= qty  # Reduce quantity

        # Commit changes
        db.session.add(transaction)
        db.session.commit()

        # Submit sell order to the API
        api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')

        return jsonify({
            "message": f"Successfully sold {qty} shares of {symbol} at ${price}!",
            "price": price
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.route('/buy', methods=['POST'])
@login_required
def buy_stock():
    try:
        data = request.get_json()
        print(data,"dataa")
        # user_id1 = data.get('user_id') or session.get('user_id')
        # print(user_id1,"lllllllllllllll")

        symbol = data.get('symbol')
        qty = int(data.get('quantity', 1))
     
        price = data.get('price')  
        print(price,"ooo")
     

        # Find the user's wallet
        token = session.get("access_token")
        print(token,"token are")

        decoded_token = decode_token(token)
        user_id = decoded_token.get("sub") 
        print("Decoded User ID:", user_id)


        print(type(price),"type")
        wallet = Wallet.query.filter_by(user_id=user_id).first()
        print(wallet)
        if not wallet:
            return jsonify({"error": "Wallet not found"}), 400
        price= float(price)
        # print(prices)


        total_cost = qty * price  # Total cost of purchase
        
        print(total_cost,"total cosr")

        # Check if the user has enough balance
        if wallet.available_balance < total_cost:
            return jsonify({"error": "Insufficient balance"}), 400

        # Record transaction before updating balance
        before_balance = wallet.available_balance
        after_balance = before_balance - total_cost

        # Create a new transaction record
        transaction = Transaction(
            wallet_id=wallet.id,
            type='buy',
            before_balance=before_balance,
            after_balance=after_balance,
            comment=f"Bought {qty} shares of {symbol} at ${price}",
            status="success"
        )

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

        # Commit changes to the database
        db.session.add(transaction)
        db.session.commit()

        # Submit buy order to the API
        api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')

        return jsonify({
            "message": f"Successfully bought {qty} shares of {symbol} at ${price}!",
            "price": price
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400