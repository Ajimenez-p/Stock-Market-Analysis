'''
Stock Market Analysis and Backtesting Strategy

This script allows users to input ticker symbols, downloads historical stock data,
calculates technical indicators, optimizes strategy parameters, generates buy/sell signals,
and evaluates the performance of a trading strategy based on technical analysis.

Results are saved to CSV files and plots are generated for visualization.

Angel Jimenez
2024-10-17
'''

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def main():
    '''
    Main function to execute the stock analysis and backtesting.
    Prompts the user for ticker symbols, processes each one, and outputs results.
    '''
    if len(sys.argv) != 3:
        start_date, end_date = '2000-10-17', '2024-10-17'
        print('No dates specified in args, using default (2000-10-17 to 2024-10-17)')
    else:
        start_date, end_date = sys.argv[1], sys.argv[2]

    tickers = get_user_tickers()

    if not tickers:
        print('No tickers entered, exiting...')
        return

    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    results = []

    for ticker in tickers:
        print(f'\nBacktesting strategy on {ticker}...\n')

        stock_data = download_stock_data(ticker, start_date, end_date)

        if stock_data is None:
            continue

        perform_exploratory_analysis(stock_data)

        # Optimize strategy parameters
        best_params = optimize_strategy(stock_data)
        rsi_lower, rsi_upper, sma_window, ema_window = best_params

        # Calculate indicators with optimized parameters
        stock_data = calculate_indicators(stock_data, sma_window, ema_window)
        stock_data = calculate_rsi(stock_data)
        stock_data = calculate_macd(stock_data)

        # Generate signals and positions
        stock_data = generate_signals(stock_data, rsi_lower, rsi_upper)
        stock_data = create_positions(stock_data)
        stock_data = calculate_returns(stock_data)

        # Plot and save cumulative returns
        plot_cumulative_returns(stock_data, ticker, output_folder)

        # Evaluate performance
        total_stock_return, total_strategy_return = evaluate_performance(stock_data)
        annualized_returns(stock_data)
        calculate_volatility(stock_data)
        calculate_sharpe_ratio(stock_data)

        # Save results
        save_results(stock_data, ticker, output_folder)

        # Store performance results
        results.append({
            'Ticker': ticker,
            'Total Stock Return': total_stock_return,
            'Total Strategy Return': total_strategy_return
        })

    # Display performance summary
    display_performance_summary(results)


def get_user_tickers():
    '''
    Prompts the user to input ticker symbols until 'stop' is entered.
    Returns:
        List of ticker symbols entered by the user.
    '''
    tickers = []
    while True:
        user_input = input('Enter a ticker symbol (or type \'stop\' to finish): ').strip()
        if user_input.lower() == 'stop':
            break
        elif user_input:
            tickers.append(user_input.upper())
    return tickers


def download_stock_data(ticker, start_date, end_date):
    '''
    Downloads historical stock data for a given ticker symbol.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    '''
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f'Error: No data found for ticker {ticker}. Skipping...')
            return None
        return data
    except Exception as e:
        print(f'Error fetching data for {ticker}: {e}')
        return None


def perform_exploratory_analysis(data):
    '''
    Performs exploratory data analysis on the stock data.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
    '''
    # Exploratory data analysis
    print('\nMissing values in dataset:')
    print(data.isnull().sum())
    print('\nSummary stats for dataset:')
    print(data.describe())
    print('\n')


def calculate_indicators(data, sma_window=20, ema_window=20):
    '''
    Calculates Simple Moving Average (SMA) and Exponential Moving Average (EMA).
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        sma_window (int): Window size for SMA.
        ema_window (int): Window size for EMA.
    Returns:
        pd.DataFrame: DataFrame with added SMA and EMA columns.
    '''
    data['SMA'] = data['Close'].rolling(window=sma_window).mean()
    data['EMA'] = data['Close'].ewm(span=ema_window, adjust=False).mean()
    return data


def calculate_rsi(data, window=14):
    '''
    Calculates the Relative Strength Index (RSI).
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        window (int): Window size for RSI calculation.
    Returns:
        pd.DataFrame: DataFrame with added RSI column.
    '''
    delta = data['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # Prevent division by zero
    data['RSI'] = 100 - (100 / (1 + rs))

    # Ensure that RSI values are between 0 and 100
    data['RSI'] = data['RSI'].clip(0, 100)

    return data


def calculate_macd(data):
    '''
    Calculates the Moving Average Convergence Divergence (MACD).
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
    Returns:
        pd.DataFrame: DataFrame with added MACD and MACD Signal columns.
    '''
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data


def generate_signals(data, rsi_lower=30, rsi_upper=70):
    '''
    Generates buy and sell signals based on RSI and MACD indicators.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        rsi_lower (int): Lower threshold for RSI buy signal.
        rsi_upper (int): Upper threshold for RSI sell signal.
    Returns:
        pd.DataFrame: DataFrame with added Signal column.
    '''
    data['Signal'] = 0

    # RSI-based signals
    data['Signal'] = np.where(data['RSI'] < rsi_lower, 1, data['Signal'])
    data['Signal'] = np.where(data['RSI'] > rsi_upper, -1, data['Signal'])

    # MACD-based signals
    macd_signal_change = np.where(data['MACD'] > data['MACD_Signal'], 1, 0)
    macd_signal_change = pd.Series(macd_signal_change, index=data.index)
    data['Signal'] = np.where(
        (data['MACD'] < data['MACD_Signal']) & (macd_signal_change.shift(1) == 1),
        -1, data['Signal']
    )
    data['Signal'] = np.where(
        (data['MACD'] > data['MACD_Signal']) & (macd_signal_change.shift(1) == 0),
        1, data['Signal']
    )

    return data


def create_positions(data):
    '''
    Creates positions based on generated signals.
    Args:
        data (pd.DataFrame): DataFrame containing stock data and signals.
    Returns:
        pd.DataFrame: DataFrame with added Position column.
    '''
    data['Position'] = data['Signal'].replace(to_replace=0, value=np.nan).ffill().shift()
    data['Position'] = data['Position'].fillna(0)
    return data


def calculate_returns(data):
    '''
    Calculates stock and strategy returns.
    Args:
        data (pd.DataFrame): DataFrame containing stock data and positions.
    Returns:
        pd.DataFrame: DataFrame with added returns columns.
    '''
    data['Stock_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Stock_Returns'] * data['Position']
    data['Cumulative_Stock_Returns'] = (1 + data['Stock_Returns']).cumprod() - 1
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod() - 1
    return data


def plot_cumulative_returns(data, ticker, output_folder):
    '''
    Plots and saves cumulative returns for the stock and strategy.
    Args:
        data (pd.DataFrame): DataFrame containing cumulative returns.
        ticker (str): Stock ticker symbol.
        output_folder (str): Directory to save the plot.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Cumulative_Stock_Returns'], label=f'{ticker} Stock Returns', color='blue')
    plt.plot(data.index, data['Cumulative_Strategy_Returns'], label='Strategy Returns', color='green')
    plt.title(f'{ticker} Cumulative Returns: Stock vs Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = os.path.join(output_folder, f'{ticker}_cumulative_returns.png')
    plt.savefig(plot_filename)
    print(f'Plot saved to {plot_filename}')

    # Display the plot
    plt.show()
    plt.close()


def evaluate_performance(data):
    '''
    Evaluates the performance of the strategy.
    Args:
        data (pd.DataFrame): DataFrame containing cumulative returns.
    Returns:
        tuple: Total stock return and total strategy return.
    '''
    total_stock_return = data['Cumulative_Stock_Returns'].iloc[-1]
    total_strategy_return = data['Cumulative_Strategy_Returns'].iloc[-1]

    print(f'Total Stock Return: {total_stock_return:.2%}')
    print(f'Total Strategy Return: {total_strategy_return:.2%}\n')

    return total_stock_return, total_strategy_return


def annualized_returns(data, periods_per_year=252):
    '''
    Calculates and prints annualized returns for the stock and strategy.
    Args:
        data (pd.DataFrame): DataFrame containing cumulative returns.
        periods_per_year (int): Number of trading periods in a year.
    '''
    total_days = len(data)
    total_stock_return = data['Cumulative_Stock_Returns'].iloc[-1]
    total_strategy_return = data['Cumulative_Strategy_Returns'].iloc[-1]

    annual_stock_return = (1 + total_stock_return) ** (periods_per_year / total_days) - 1
    annual_strategy_return = (1 + total_strategy_return) ** (periods_per_year / total_days) - 1

    print(f'Annualized Stock Return: {annual_stock_return:.2%}')
    print(f'Annualized Strategy Return: {annual_strategy_return:.2%}\n')


def calculate_volatility(data, periods_per_year=252):
    '''
    Calculates and prints the annualized volatility for the stock and strategy.
    Args:
        data (pd.DataFrame): DataFrame containing returns.
        periods_per_year (int): Number of trading periods in a year.
    '''
    stock_volatility = data['Stock_Returns'].std() * np.sqrt(periods_per_year)
    strategy_volatility = data['Strategy_Returns'].std() * np.sqrt(periods_per_year)

    print(f'Stock Volatility (Annualized): {stock_volatility:.2%}')
    print(f'Strategy Volatility (Annualized): {strategy_volatility:.2%}\n')


def calculate_sharpe_ratio(data, risk_free_rate=0.01, periods_per_year=252):
    '''
    Calculates and prints the Sharpe Ratio for the stock and strategy.
    Args:
        data (pd.DataFrame): DataFrame containing returns.
        risk_free_rate (float): Risk-free interest rate.
        periods_per_year (int): Number of trading periods in a year.
    '''
    excess_stock_return = data['Stock_Returns'] - risk_free_rate / periods_per_year
    excess_strategy_return = data['Strategy_Returns'] - risk_free_rate / periods_per_year

    stock_sharpe_ratio = (excess_stock_return.mean() / excess_stock_return.std()) * np.sqrt(periods_per_year)
    strategy_sharpe_ratio = (excess_strategy_return.mean() / excess_strategy_return.std()) * np.sqrt(periods_per_year)

    print(f'Stock Sharpe Ratio: {stock_sharpe_ratio:.2f}')
    print(f'Strategy Sharpe Ratio: {strategy_sharpe_ratio:.2f}\n')


def save_results(data, ticker, output_folder):
    '''
    Saves the analysis results to a CSV file.
    Args:
        data (pd.DataFrame): DataFrame containing analysis results.
        ticker (str): Stock ticker symbol.
        output_folder (str): Directory to save the CSV file.
    '''
    try:
        output_filename = os.path.join(output_folder, f'{ticker}_stock_analysis.csv')
        data.to_csv(output_filename)
        print(f'Data saved to {output_filename}')
    except Exception as e:
        print(f'Error saving CSV for {ticker}: {e}')


def display_performance_summary(results):
    '''
    Displays a summary of the performance for all tickers.
    Args:
        results (list): List of dictionaries containing performance metrics.
    '''
    print('\nPerformance Summary for All Stocks:')
    for result in results:
        ticker = result['Ticker']
        total_stock_return = result['Total Stock Return']
        total_strategy_return = result['Total Strategy Return']
        print(f'Ticker: {ticker}, Total Stock Return: {total_stock_return:.2%}, '
              f'Total Strategy Return: {total_strategy_return:.2%}')


def optimize_strategy(data):
    '''
    Optimizes the trading strategy parameters.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
    Returns:
        tuple: Best parameters found (rsi_lower, rsi_upper, sma_window, ema_window).
    '''
    rsi_lower_range = [20, 25, 30]
    rsi_upper_range = [70, 75, 80]
    sma_window_range = [10, 20, 50]
    ema_window_range = [12, 20, 50]

    best_params = None
    best_performance = -np.inf

    for rsi_lower in rsi_lower_range:
        for rsi_upper in rsi_upper_range:
            for sma_window in sma_window_range:
                for ema_window in ema_window_range:
                    temp_data = data.copy()
                    temp_data = calculate_indicators(temp_data, sma_window, ema_window)
                    temp_data = calculate_rsi(temp_data)
                    temp_data = calculate_macd(temp_data)
                    temp_data = generate_signals(temp_data, rsi_lower, rsi_upper)
                    temp_data = create_positions(temp_data)
                    temp_data = calculate_returns(temp_data)

                    total_strategy_return = temp_data['Cumulative_Strategy_Returns'].iloc[-1]

                    if total_strategy_return > best_performance:
                        best_performance = total_strategy_return
                        best_params = (rsi_lower, rsi_upper, sma_window, ema_window)

    print(f'Best Parameters: RSI Lower: {best_params[0]}, RSI Upper: {best_params[1]}, '
          f'SMA Window: {best_params[2]}, EMA Window: {best_params[3]}\n')
    return best_params


if __name__ == '__main__':
    main()
