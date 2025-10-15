"""
Task 1: Target Engineering & Return Calculation

This module implements functions for calculating forward returns and analyzing return distributions
for the quantitative strategy development project.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_forward_returns(
    data: pd.DataFrame, 
    forward_periods: int = 12,
    price_column: str = 'close_px'
) -> pd.DataFrame:
    """
    Calculate forward returns for each stock at every 5-minute interval.
    
    This function computes the forward return over a specified number of periods
    (default 12 periods = 1 hour for 5-minute data) for each stock in the dataset.
    The forward return is calculated as (future_price / current_price - 1).
    
    Args:
        data (pd.DataFrame): Multi-level DataFrame with stock data
                           Expected structure: MultiIndex columns (stock_symbol, data_fields)
                           Index should be timestamp
        forward_periods (int): Number of periods to look forward (default 12 for 1-hour)
        price_column (str): Column name for price data (default 'close_px')
    
    Returns:
        pd.DataFrame: DataFrame with forward returns for each stock
                     Columns: stock symbols
                     Index: timestamps
                     
    Example:
        >>> data = load_5min_data()  # Load your 5-minute data
        >>> forward_returns = calculate_forward_returns(data, forward_periods=12)
        >>> print(forward_returns.head())
    
    Notes:
        - The last 'forward_periods' rows will contain NaN values
        - Returns are calculated as simple returns (not log returns)
        - Data should be properly sorted by timestamp before calling this function
    """
    
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Input data must have MultiIndex columns (stock_symbol, data_fields)")
    
    # 2. Extract all stock symbols
    stock_symbols = data.columns.get_level_values(0).unique()
    
    # Dictionary to store forward returns for each stock
    forward_returns_dict = {}
    
    # 3. For each stock:
    for stock in stock_symbols:
        try:
            # Get current prices for this stock
            current_prices = data[(stock, price_column)]
            
            # Get future prices by shifting forward
            future_prices = current_prices.shift(-forward_periods)
            
            # Calculate forward returns: (future_price / current_price) - 1
            forward_returns = (future_prices / current_prices) - 1
            
            # Store in dictionary
            forward_returns_dict[stock] = forward_returns
            
        except KeyError:
            print(f"Warning: Price column '{price_column}' not found for stock {stock}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing stock {stock}: {e}")
            continue
    
    # 4. Combine all stock returns into DataFrame
    forward_returns_df = pd.DataFrame(forward_returns_dict)
    forward_returns_df.index = data.index  # Ensure index is preserved
    
    return forward_returns_df


def calculate_weekly_returns(daily_data: pd.DataFrame, price_column: str = 'close_px') -> pd.DataFrame:
    """
    Calculate weekly returns for each stock using daily K-bar data.
    """
    if not isinstance(daily_data.columns, pd.MultiIndex):
        raise ValueError("Input data must have MultiIndex columns (stock_symbol, data_fields)")
    
    stock_symbols = daily_data.columns.get_level_values(0).unique()
    weekly_returns_dict = {}
    
    for stock in stock_symbols:
        try:
            # Get daily price data
            stock_prices = daily_data[(stock, price_column)]
            
            # 移除NaN，然后确保数据按时间排序
            stock_prices_clean = stock_prices.dropna().sort_index()
            
            if len(stock_prices_clean) < 8:  # 至少需要2周的数据
                continue
            
            # 使用'W-FRI'确保每周五结束
            weekly_prices = stock_prices_clean.resample('W-FRI').last()
            
            # 计算周度收益
            stock_weekly_returns = weekly_prices.pct_change()
            
            # 关键修改：移除第一个NaN值
            stock_weekly_returns = stock_weekly_returns.iloc[1:]  # 移除第一行NaN
            
            weekly_returns_dict[stock] = stock_weekly_returns
            
        except KeyError:
            print(f"Warning: Price column '{price_column}' not found for stock {stock}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing stock {stock}: {e}")
            continue
    
    weekly_returns_df = pd.DataFrame(weekly_returns_dict)
    return weekly_returns_df


def plot_return_distribution(
    returns_data: pd.DataFrame, 
    sample_stocks: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Dict[str, Dict[str, float]]:
    """
    Visualize and analyze the distribution of returns for sample stocks.
    """
# 1. Verify sample stocks exist in the data
    available_stocks = [stock for stock in sample_stocks if stock in returns_data.columns]
    missing_stocks = [stock for stock in sample_stocks if stock not in returns_data.columns]
    
    if missing_stocks:
        print(f"Warning: The following stocks are not in the data and will be skipped: {missing_stocks}")
    
    if not available_stocks:
        raise ValueError("None of the specified sample stocks exist in the returns data")
    
    n_stocks = len(available_stocks)
    
    # Create figure with 2×n_stocks subplot layout
    fig, axes = plt.subplots(2, n_stocks, figsize=figsize)
    if n_stocks == 1:
        axes = axes.reshape(2, 1)
    
    # Dictionary to store statistics
    statistics_dict = {}
    
    # 3. Calculate statistics for each stock:
    for i, stock in enumerate(available_stocks):
        # Get returns data for the stock, remove NaN values
        stock_returns = returns_data[stock].dropna()
        
        if len(stock_returns) == 0:
            print(f"Warning: No valid returns data for stock {stock}. Skipping.")
            continue
        
        # 计算所有必需的统计量
        count = len(stock_returns)  # 数据点数量
        mean_return = stock_returns.mean()
        std_return = stock_returns.std()
        skewness = stock_returns.skew()
        kurtosis_val = stock_returns.kurtosis()  # 超额峰度
        
        # 正态性检验
        jarque_bera_pvalue = np.nan
        if count >= 8:
            try:
                jb_stat, jarque_bera_pvalue = stats.jarque_bera(stock_returns)
            except:
                jarque_bera_pvalue = np.nan
        
        # 存储统计量 - 确保包含count
        statistics_dict[stock] = {
            'count': float(count),  # 转换为float确保类型一致
            'mean': float(mean_return),
            'std': float(std_return),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis_val + 3),  # 转换为原始峰度
            'jarque_bera_pvalue': float(jarque_bera_pvalue) if not np.isnan(jarque_bera_pvalue) else np.nan
        }
        
        # 4. Create visualizations:
        # First row: histogram + normal distribution overlay
        ax_hist = axes[0, i]
        
        # Create histogram
        n, bins, patches = ax_hist.hist(stock_returns, bins=50, density=True, alpha=0.7, 
                                       color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        xmin, xmax = ax_hist.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        normal_pdf = stats.norm.pdf(x, mean_return, std_return)
        ax_hist.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal Distribution')
        
        # Customize histogram subplot
        ax_hist.set_title(f'{stock}\nReturn Distribution', fontsize=12, fontweight='bold')
        ax_hist.set_xlabel('Returns')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Add statistical information as text box
        stats_text = f'Count: {count}\nMean: {mean_return:.6f}\nStd: {std_return:.6f}\nSkew: {skewness:.4f}\nKurtosis: {kurtosis_val+3:.4f}'
        if not np.isnan(jarque_bera_pvalue):
            stats_text += f'\nJB p-value: {jarque_bera_pvalue:.4f}'
        ax_hist.text(0.05, 0.95, stats_text, transform=ax_hist.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Second row: Q-Q plot
        ax_qq = axes[1, i]
        
        # Create Q-Q plot
        stats.probplot(stock_returns, dist="norm", plot=ax_qq)
        ax_qq.set_title(f'{stock}\nQ-Q Plot', fontsize=12, fontweight='bold')
        ax_qq.grid(True, alpha=0.3)
        
        # Add normality assessment to Q-Q plot
        if not np.isnan(jarque_bera_pvalue):
            normality_status = "Non-Normal" if jarque_bera_pvalue < 0.05 else "Normal"
            color = 'lightcoral' if jarque_bera_pvalue < 0.05 else 'lightgreen'
            ax_qq.text(0.05, 0.95, f'Normality: {normality_status}', 
                      transform=ax_qq.transAxes, fontsize=10, fontweight='bold',
                      verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # 6. Save plots (if path specified)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return statistics_dict
    

def analyze_return_properties(returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze statistical properties of returns for all stocks.
    """
        # Handle edge cases with empty data
    if returns_data.empty:
        print("Warning: Input returns data is empty")
        return pd.DataFrame()
    
    # List to store statistics for each stock
    stats_list = []
    
    # 1. Calculate statistics for each stock's return series
    for stock in returns_data.columns:
        stock_returns = returns_data[stock].dropna()
        
        # Skip if no valid data
        if len(stock_returns) == 0:
            print(f"Warning: No valid returns data for stock {stock}. Skipping.")
            continue
        
        # Basic statistics
        count = len(stock_returns)
        mean = stock_returns.mean()
        std = stock_returns.std()
        minimum = stock_returns.min()
        maximum = stock_returns.max()
        
        # Quantiles - 关键修改：使用 q50 而不是 median
        q25 = stock_returns.quantile(0.25)
        q50 = stock_returns.quantile(0.50)  # median
        q75 = stock_returns.quantile(0.75)
        
        # Distribution shape
        skewness = stock_returns.skew()
        kurtosis = stock_returns.kurtosis()  # Excess kurtosis (normal = 0)
        
        # 2. Perform normality test (when sample size >= 8)
        jarque_bera_pvalue = np.nan
        is_normal = np.nan
        
        if count >= 8:
            try:
                jb_stat, jarque_bera_pvalue = stats.jarque_bera(stock_returns)
                is_normal = jarque_bera_pvalue > 0.05
            except:
                jarque_bera_pvalue = np.nan
                is_normal = np.nan
        
        # 可选：移除额外的统计量以匹配测试期望
        # volatility_annualized = std * np.sqrt(252)  # 如果测试不需要，可以注释掉
        # sharpe_ratio = mean / std * np.sqrt(252) if std != 0 else np.nan
        
        # Store statistics for this stock - 关键修改：使用 q50
        stock_stats = {
            'count': count,
            'mean': mean,
            'std': std,
            'min': minimum,
            'max': maximum,
            'q25': q25,
            'q50': q50,  # 修改：从 'median' 改为 'q50'
            'q75': q75,
            'skewness': skewness,
            'kurtosis': kurtosis + 3,  # Convert to raw kurtosis (normal = 3)
            'jarque_bera_pvalue': jarque_bera_pvalue,
            'is_normal': is_normal,
            # 如果测试不需要这些，可以移除
            # 'volatility_annualized': volatility_annualized,
            # 'sharpe_ratio': sharpe_ratio
        }
        
        stats_list.append((stock, stock_stats))
    
    # 3. Organize results into DataFrame with stock symbols as index
    if not stats_list:
        print("Warning: No valid statistics calculated for any stock")
        return pd.DataFrame()
    
    # Create DataFrame
    summary_df = pd.DataFrame(
        [stats for _, stats in stats_list],
        index=[stock for stock, _ in stats_list]
    )
    
    return summary_df


# Example usage and testing functions
def main():
    """
    Main function to demonstrate the usage of return calculation functions.
    
    This function provides examples of how to use the implemented functions
    and can be used for testing purposes.
    """
    print("Task 1: Target Engineering & Return Calculation")
    print("=" * 50)
    
    # Note: This is a template - actual data loading will depend on your data structure
    print("To use these functions:")
    print("1. Load your 5-minute data using the data loader")
    print("2. Calculate forward returns using calculate_forward_returns()")
    print("3. Load daily data and calculate weekly returns")
    print("4. Analyze return distributions for sample stocks")
    
    print("\nExample code:")
    print("""
    # Load data (implement based on your data structure)
    # data_5min = load_5min_data()
    # data_daily = load_daily_data()
    
    # Calculate forward returns
    # forward_returns = calculate_forward_returns(data_5min, forward_periods=12)
    
    # Calculate weekly returns
    # weekly_returns = calculate_weekly_returns(data_daily)
    
    # Analyze distributions
    # sample_stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
    # stats = plot_return_distribution(forward_returns, sample_stocks)
    """)


if __name__ == "__main__":
    main()


