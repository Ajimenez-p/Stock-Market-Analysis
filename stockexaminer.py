#Data manip., visualization, and analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_ta as ta

#fetches stock data from Yahoo!
import yfinance as yf 

def main():
    #start plt instance
    #%matplotlib inline
    #plt.style.use('seaborn-darkgrid')

    #Grab data from Yahoo! Finance
    ticker = 'HD' #The Home Depot
    start_date = '2021-07-21'
    end_date = '2024-10-11'
    data = yf.download(ticker, start=start_date, end=end_date)
    
    #Display first few rows of dataset
    print(f'First few rows of {ticker} dataset:')
    print(data.head())

    #Exploratory data analysis
    print('\nMissing values in dataset:')
    print(data.isnull().sum()) #
    print('\nSummary stats for dataset:')
    print(data.describe())

    #Calculate SMA, EMA
    data['SMA_20'] = data['Close'].rolling(window=20).mean() # 20 Day SMA
    data['SMA_50'] = data['Close'].rolling(window=50).mean() # 50 Day SMA
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean() #20 day EMA

    #Calculate RSI
    delta = data['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI_14'] = 100 - (100/ (1+rs)) #RSI Formula

     # 4. Calculate MACD (12-day EMA - 26-day EMA)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()  # Signal line

    # Display the technical indicators
    print("\nTechnical Indicators (Last 5 rows):")
    print(data[['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal']].tail())

    # 5. Plot Closing Price and Moving Averages
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Closing Price', color='blue')
    plt.plot(data.index, data['SMA_20'], label='SMA 20', linestyle='--', color='green')
    plt.plot(data.index, data['SMA_50'], label='SMA 50', linestyle='--', color='red')
    plt.title(f'{ticker} Closing Price with SMAs')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6. Plot RSI
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data['RSI_14'], label='RSI (14)', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title(f'{ticker} RSI (14)')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 7. Plot MACD
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['MACD'], label='MACD', color='blue')
    plt.plot(data.index, data['MACD_Signal'], label='Signal Line', color='red')
    plt.title(f'{ticker} MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
