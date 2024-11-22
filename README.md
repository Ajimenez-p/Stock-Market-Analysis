# Stock Market Analysis and Backtesting Strategy

## Overview

This project is a comprehensive stock market analysis tool and backtesting strategy simulator. It allows users to input ticker symbols, download historical stock data, calculate technical indicators, optimize strategy parameters, generate buy/sell signals, and evaluate the performance of a trading strategy based on technical analysis.

**Key features include:**

- **Data Retrieval**: Fetches historical stock data using the `yfinance` API.
- **Technical Indicators**: Calculates popular indicators such as SMA, EMA, RSI, and MACD.
- **Strategy Optimization**: Optimizes trading strategy parameters for best performance.
- **Signal Generation**: Generates buy/sell signals based on optimized indicators.
- **Backtesting**: Simulates the trading strategy over historical data.
- **Performance Evaluation**: Calculates returns, volatility, Sharpe ratio, and provides a performance summary.
- **Visualization**: Generates and saves plots of cumulative returns.
- **Data Export**: Saves analysis results to CSV files for further examination.

## Libraries Used

- **[os](https://docs.python.org/3/library/os.html)**: Interacts with the operating system to handle file paths and directories.
- **[sys](https://docs.python.org/3/library/sys.html)**: Accesses command-line arguments and system-specific parameters.
- **[matplotlib.pyplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)**: Plots graphs and charts for data visualization.
- **[numpy](https://numpy.org/)**: Performs numerical computations efficiently.
- **[pandas](https://pandas.pydata.org/)**: Manipulates and analyzes data in DataFrames.
- **[yfinance](https://pypi.org/project/yfinance/)**: Downloads historical market data from Yahoo Finance.

## Skills Gained

- **Data Retrieval and Processing**: Learned to fetch and handle large datasets from external APIs.
- **Technical Analysis**: Implemented and understood technical indicators like SMA, EMA, RSI, and MACD.
- **Strategy Optimization**: Developed methods to optimize trading strategies using parameter sweeps.
- **Backtesting**: Gained experience in simulating trading strategies over historical data to evaluate performance.
- **Data Visualization**: Enhanced skills in plotting and interpreting financial data trends.
- **Performance Evaluation**: Calculated key performance metrics including returns, volatility, and Sharpe ratio.
- **File Handling and Data Storage**: Managed data export to CSV and saving plots to files.
- **Python Programming**: Strengthened programming skills including function definitions, control flow, error handling, and code modularity.
- **Use of Data Science Libraries**: Utilized popular libraries such as NumPy, pandas, and Matplotlib for data analysis.

## Reasoning Behind the Code

The code is structured into modular functions, each responsible for a specific task, enhancing readability and maintainability. Here's the reasoning behind key components:

### User Interaction

Allows dynamic input of ticker symbols, making the tool flexible for different stocks.

```python
def get_user_tickers():
    """
    Prompts the user to input ticker symbols until 'stop' is entered.
    Returns:
        List of ticker symbols entered by the user.
    """
    tickers = []
    while True:
        user_input = input("Enter a ticker symbol (or type 'stop' to finish): ").strip()
        if user_input.lower() == 'stop':
            break
        elif user_input:
            tickers.append(user_input.upper())
    return tickers
```

### Data Download

Uses `yfinance` to fetch historical data, with error handling to manage unavailable data.

```python
def download_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock data for a given ticker symbol.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f'Error: No data found for ticker {ticker}. Skipping...')
            return None
        return data
    except Exception as e:
        print(f'Error fetching data for {ticker}: {e}')
        return None
```

### Technical Indicators Calculation

Computes SMA, EMA, RSI, and MACD to inform trading signals.

```python
def calculate_indicators(data, sma_window=20, ema_window=20):
    """
    Calculates Simple Moving Average (SMA) and Exponential Moving Average (EMA).
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        sma_window (int): Window size for SMA.
        ema_window (int): Window size for EMA.
    Returns:
        pd.DataFrame: DataFrame with added SMA and EMA columns.
    """
    data['SMA'] = data['Close'].rolling(window=sma_window).mean()
    data['EMA'] = data['Close'].ewm(span=ema_window, adjust=False).mean()
    return data
```

### Signal Generation

Uses indicators to generate buy/sell signals based on defined thresholds.

```python
def generate_signals(data, rsi_lower=30, rsi_upper=70):
    """
    Generates buy and sell signals based on RSI and MACD indicators.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        rsi_lower (int): Lower threshold for RSI buy signal.
        rsi_upper (int): Upper threshold for RSI sell signal.
    Returns:
        pd.DataFrame: DataFrame with added Signal column.
    """
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
```

### Strategy Optimization

Iterates over ranges of parameters to find the optimal strategy settings.

```python
def optimize_strategy(data):
    """
    Optimizes the trading strategy parameters.
    Args:
        data (pd.DataFrame): DataFrame containing stock data.
    Returns:
        tuple: Best parameters found (rsi_lower, rsi_upper, sma_window, ema_window).
    """
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
```

### Performance Metrics

Calculates cumulative returns, annualized returns, volatility, and Sharpe ratio to evaluate strategy effectiveness.

```python
def calculate_sharpe_ratio(data, risk_free_rate=0.01, periods_per_year=252):
    """
    Calculates and prints the Sharpe Ratio for the stock and strategy.
    Args:
        data (pd.DataFrame): DataFrame containing returns.
        risk_free_rate (float): Risk-free interest rate.
        periods_per_year (int): Number of trading periods in a year.
    """
    excess_stock_return = data['Stock_Returns'] - risk_free_rate / periods_per_year
    excess_strategy_return = data['Strategy_Returns'] - risk_free_rate / periods_per_year

    stock_sharpe_ratio = (excess_stock_return.mean() / excess_stock_return.std()) * np.sqrt(periods_per_year)
    strategy_sharpe_ratio = (excess_strategy_return.mean() / excess_strategy_return.std()) * np.sqrt(periods_per_year)

    print(f'Stock Sharpe Ratio: {stock_sharpe_ratio:.2f}')
    print(f'Strategy Sharpe Ratio: {strategy_sharpe_ratio:.2f}\n')
```

### Visualization and Export

Plots cumulative returns and saves results to files for reporting and further analysis.

```python
def plot_cumulative_returns(data, ticker, output_folder):
    """
    Plots and saves cumulative returns for the stock and strategy.
    Args:
        data (pd.DataFrame): DataFrame containing cumulative returns.
        ticker (str): Stock ticker symbol.
        output_folder (str): Directory to save the plot.
    """
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
```

## Important Skills for the Modern Age

This project highlights several important skills relevant in today's data-driven and technologically advanced world:

- **Financial Data Analysis**: Ability to analyze and interpret financial data is crucial in finance and investing sectors.
- **Algorithmic Trading Strategies**: Understanding and developing algorithmic strategies is valuable for modern trading and investment management.
- **Data Visualization**: Proficiency in visualizing data to uncover insights and communicate findings effectively.
- **Backtesting Techniques**: Skills in simulating strategies on historical data to predict future performance.
- **Python Programming**: Mastery of Python, a leading programming language in data science, finance, and technology industries.
- **Use of Data Science Libraries**: Expertise in leveraging libraries like pandas, NumPy, and Matplotlib for efficient data analysis.
- **Optimization Methods**: Applying optimization techniques to enhance model and strategy performance.
- **API Integration**: Experience in integrating external data sources via APIs, a common requirement in many tech roles.
- **Critical Thinking and Problem Solving**: Demonstrated ability to design, implement, and refine complex systems.

## How to Use the Project

1. **Install Required Libraries**: Ensure you have the necessary Python libraries installed:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script**: Execute the script using Python:

   ```bash
   python stock_analysis.py
   ```

   Optionally, you can specify start and end dates as command-line arguments:

   ```bash
   python stock_analysis.py 2000-01-01 2024-10-17
   ```

3. **Input Ticker Symbols**: When prompted, enter the stock ticker symbols you're interested in analyzing. Type `'stop'` to finish input.

4. **View Results**: The script will output analysis, generate plots, and save results to the `output` folder.

## Example Output

Here's an example of the cumulative returns plot generated by the script:

![Cumulative Returns Plot](output/HD_cumulative_returns.png)

## Conclusion

This project serves as a robust tool for stock market analysis and trading strategy development. It integrates data retrieval, technical analysis, optimization, and performance evaluation, embodying key skills and technologies pertinent in finance and data science domains.
