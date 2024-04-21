from flask import Flask, request, jsonify

app = Flask(__name__)

# Fictional stock data for FAANG companies
def getStockPrice(stockName):
    """Get the current stock price for a given company."""
    stockName = stockName.lower()
    if "meta" in stockName:
        return {"stock_id": "1", "companyName": "Meta", "stockSymbol": "FB", "price": 340.5, "currency": "USD", "exchange": "NASDAQ", "marketCap": 1000000000000}
    elif "alphabet" in stockName:
        return {"stock_id": "2", "companyName": "Alphabet", "stockSymbol": "GOOG", "price": 2800.0, "currency": "USD", "exchange": "NASDAQ", "marketCap": 2000000000000}
    elif "amazon" in stockName:
        return {"stock_id": "3", "companyName": "Amazon", "stockSymbol": "AMZN", "price": 3200.0, "currency": "USD", "exchange": "NASDAQ", "marketCap": 1500000000000}
    elif "apple" in stockName:
        return {"stock_id": "4", "companyName": "Apple", "stockSymbol": "AAPL", "price": 145.0, "currency": "USD", "exchange": "NASDAQ", "marketCap": 2000000000000}
    elif "netflix" in stockName:
        return {"stock_id": "5", "companyName": "Netflix", "stockSymbol": "NFLX", "price": 500.0, "currency": "USD", "exchange": "NASDAQ", "marketCap": 250000000000}
    else:
        return {"stockName": stockName, "price": "unknown"}

@app.route('/price', methods=['GET'])
def price():
    stockName = request.args.get('stockName', '')
    if not stockName:
        return jsonify({'error': 'Missing stock name'}), 400
    price = getStockPrice(stockName)
    return jsonify(price)

if __name__ == '__main__':
    app.run(debug=True)