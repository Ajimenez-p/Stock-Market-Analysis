import pytest
import pandas as pd
import numpy as np
from stockexaminer import (
    get_user_tickers, download_stock_data, perform_exploratory_analysis, calculate_indicators,
    calculate_rsi, calculate_macd, generate_signals, create_positions, calculate_returns,
    evaluate_performance, annualized_returns, calculate_volatility, calculate_sharpe_ratio,
    save_results, optimize_strategy
)

# Sample test data
mock_data = pd.DataFrame({
    'Close': [150, 152, 154, 153, 155, 157, 159, 160, 162, 161, 163, 164, 165, 167, 168],
    'Open': [148, 150, 152, 151, 153, 156, 158, 159, 161, 160, 162, 163, 164, 166, 167],
    'High': [151, 153, 155, 154, 157, 158, 160, 161, 163, 162, 164, 165, 166, 168, 169],
    'Low': [147, 149, 150, 151, 152, 155, 157, 158, 160, 159, 161, 162, 163, 165, 166],
    'Volume': [1000, 1100, 1050, 1020, 1010, 1040, 1060, 1030, 1080, 1070, 1090, 1100, 1110, 1130, 1140]
})

# Test perform_exploratory_analysis
def test_perform_exploratory_analysis(capsys):
    perform_exploratory_analysis(mock_data)
    captured = capsys.readouterr()
    assert 'Missing values in dataset:' in captured.out, 'Expected missing values output.'
    assert 'Summary stats for dataset:' in captured.out, 'Expected summary stats output.'


# Test calculate_indicators
def test_calculate_indicators():
    result = calculate_indicators(mock_data, sma_window=3, ema_window=3)
    assert 'SMA' in result.columns, 'SMA column should be present.'
    assert 'EMA' in result.columns, 'EMA column should be present.'
    assert not result['SMA'].isnull().all(), 'SMA values should be calculated.'
    assert not result['EMA'].isnull().all(), 'EMA values should be calculated.'


# Test calculate_rsi
def test_calculate_rsi():
    result = calculate_rsi(mock_data)
    # Assert that the RSI column exists
    assert 'RSI' in result.columns, 'RSI column should be present.'
    # Assert that RSI values after the initial NaN are valid
    assert result['RSI'][13:].notnull().all(), 'RSI values should be calculated after 13th row.'
    # Ensure all calculated RSI values are between 0 and 100
    assert result['RSI'][13:].between(0, 100).all(), 'RSI values should be between 0 and 100.'



# Test calculate_macd
def test_calculate_macd():
    result = calculate_macd(mock_data)
    assert 'MACD' in result.columns, 'MACD column should be present.'
    assert 'MACD_Signal' in result.columns, 'MACD Signal column should be present.'
    assert not result['MACD'].isnull().all(), 'MACD values should be calculated.'


# Test generate_signals
def test_generate_signals():
    result = generate_signals(mock_data, rsi_lower=30, rsi_upper=70)
    assert 'Signal' in result.columns, 'Signal column should be present.'
    assert result['Signal'].isin([-1, 0, 1]).all(), 'Signals should be -1, 0, or 1.'


# Test create_positions
def test_create_positions():
    signals = generate_signals(mock_data, rsi_lower=30, rsi_upper=70)
    positions = create_positions(signals)
    assert 'Position' in positions.columns, 'Position column should be present.'
    assert positions['Position'].isin([0, 1, -1]).all(), 'Positions should be -1, 0, or 1.'


# Test calculate_returns
def test_calculate_returns():
    signals = generate_signals(mock_data, rsi_lower=30, rsi_upper=70)
    positions = create_positions(signals)
    returns = calculate_returns(positions)
    assert 'Stock_Returns' in returns.columns, 'Stock returns column should be present.'
    assert 'Strategy_Returns' in returns.columns, 'Strategy returns column should be present.'


# Test evaluate_performance
def test_evaluate_performance():
    signals = generate_signals(mock_data, rsi_lower=30, rsi_upper=70)
    positions = create_positions(signals)
    returns = calculate_returns(positions)
    stock_return, strategy_return = evaluate_performance(returns)
    assert isinstance(stock_return, float), 'Stock return should be a float.'
    assert isinstance(strategy_return, float), 'Strategy return should be a float.'


# Test annualized_returns
def test_annualized_returns(capsys):
    signals = generate_signals(mock_data, rsi_lower=30, rsi_upper=70)
    positions = create_positions(signals)
    returns = calculate_returns(positions)
    annualized_returns(returns)
    captured = capsys.readouterr()
    assert 'Annualized Stock Return:' in captured.out, 'Expected annualized stock return output.'
    assert 'Annualized Strategy Return:' in captured.out, 'Expected annualized strategy return output.'


# Test calculate_volatility
def test_calculate_volatility(capsys):
    signals = generate_signals(mock_data, rsi_lower=30, rsi_upper=70)
    positions = create_positions(signals)
    returns = calculate_returns(positions)
    calculate_volatility(returns)
    captured = capsys.readouterr()
    assert 'Stock Volatility (Annualized):' in captured.out, 'Expected stock volatility output.'
    assert 'Strategy Volatility (Annualized):' in captured.out, 'Expected strategy volatility output.'


# Test calculate_sharpe_ratio
def test_calculate_sharpe_ratio(capsys):
    signals = generate_signals(mock_data, rsi_lower=30, rsi_upper=70)
    positions = create_positions(signals)
    returns = calculate_returns(positions)
    calculate_sharpe_ratio(returns)
    captured = capsys.readouterr()
    assert 'Stock Sharpe Ratio:' in captured.out, 'Expected Sharpe ratio output.'
    assert 'Strategy Sharpe Ratio:' in captured.out, 'Expected strategy Sharpe ratio output.'


# Test optimize_strategy
def test_optimize_strategy():
    best_params = optimize_strategy(mock_data)
    assert isinstance(best_params, tuple), 'Best parameters should be a tuple.'
    assert len(best_params) == 4, 'Best parameters should contain four values (RSI lower, RSI upper, SMA window, EMA window).'


# Test save_results
def test_save_results(tmpdir):
    # Expect the file to be named with ticker 'AAPL' in the test
    expected_filename = tmpdir.join('AAPL_stock_analysis.csv')
    save_results(mock_data, 'AAPL', tmpdir)
    
    assert expected_filename.check(file=1), 'CSV file should be saved successfully.'

