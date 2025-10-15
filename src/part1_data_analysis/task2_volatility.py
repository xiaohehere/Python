"""
Task 2: Market & Asset Characterization

This module implements functions for volatility analysis and market index construction
for the quantitative strategy development project.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def calculate_rolling_volatility(
    daily_returns: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation) for stock returns.
    
    This function computes the rolling standard deviation of daily returns
    for each stock over a specified window period. Volatility can be annualized
    by multiplying by sqrt(252) for daily data.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns for each stock
                                    Columns: stock symbols
                                    Index: dates
        window (int): Rolling window size in days (default 20 for monthly volatility)
        annualize (bool): Whether to annualize volatility (default True)
    
    Returns:
        pd.DataFrame: DataFrame with rolling volatility for each stock
                     Same structure as input but with volatility values
                     
    Example:
        # >>> daily_returns = calculate_daily_returns(data)
        # >>> volatility = calculate_rolling_volatility(daily_returns, window=20)
        # >>> print(volatility.head())
    
    Notes:
        - Annualized volatility assumes 252 trading days per year
        - The first (window-1) rows will contain NaN values
        - Volatility is calculated as the standard deviation of returns
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Use pandas rolling() function to calculate rolling standard deviation:
    #    - daily_returns.rolling(window=window).std()
    # 2. If annualize=True, annualize the results:
    #    - Multiply by np.sqrt(252), assuming 252 trading days per year
    # 3. Note that the first (window-1) rows will be NaN, which is normal
    #
    # Expected output: DataFrame with same structure as input, values are rolling volatilities

    #Step 1:Use pandas rolling() function to calculate rolling standard deviation
    rolling_vol = daily_returns.rolling(window).std()

    #Step 2:If annualize=True, annualize the results:
    #    - Multiply by np.sqrt(252), assuming 252 trading days per year
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(252)

    #Step 3: Return results
    return rolling_vol


def build_equal_weight_index(
    stock_returns: pd.DataFrame,
    rebalance_freq: str = 'D'
) -> pd.Series:
    """
    Create an equal-weighted index from individual stock returns.
    
    This function constructs a market index by equally weighting all stocks
    and rebalancing at the specified frequency. The index serves as a
    benchmark for strategy performance evaluation.
    
    Args:
        stock_returns (pd.DataFrame): DataFrame with returns for each stock
                                    Columns: stock symbols
                                    Index: timestamps
        rebalance_freq (str): Rebalancing frequency ('D' for daily, 'W' for weekly)
    
    Returns:
        pd.Series: Equal-weighted index returns
                  Index: timestamps
                  Values: index returns
                  
    Example:
        # >>> daily_returns = calculate_daily_returns(data)
        # >>> ew_index = build_equal_weight_index(daily_returns)
        # >>> print(f"Index average return: {ew_index.mean():.4f}")
    
    Notes:
        - Equal weighting means each stock gets 1/N weight where N is number of stocks
        - Missing data is handled by excluding stocks with NaN returns for that period
        - Index is rebalanced at specified frequency to maintain equal weights
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Calculate equal-weight returns (simple average of all stocks):
    #    - Use stock_returns.mean(axis=1, skipna=True)
    # 2. axis=1 means calculate across rows (average across stocks)
    # 3. skipna=True automatically handles missing values, only calculates average for valid data
    # 4. Returns a time series (pd.Series)
    #
    # Expected output: Series with time index and equal-weight index returns as values

    # Step 1: Calculate equal-weight returns (simple average of all stocks)
    #   - Use stock_returns.mean(axis=1, skipna=True)
    ew_returns = stock_returns.mean(axis=1, skipna=True)

    # Step 2: Handle rebalancing frequency
    #   - If rebalance_freq='D', keep daily average
    #   - If rebalance_freq='W', resample to weekly and take mean
    if rebalance_freq == 'W':
        ew_returns = ew_returns.resample('W').mean()

    # Step 3: Return result as pd.Series
    if stock_returns.shape[1] == 1:
        ew_returns.name = stock_returns.columns[0]
    else:
        ew_returns.name = None

    return ew_returns


def plot_volatility_analysis(
    daily_returns: pd.DataFrame,
    sample_stocks: List[str],
    equal_weight_index: pd.Series,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Create comprehensive volatility analysis plots.
    
    This function generates multiple visualizations to analyze volatility patterns:
    1. Rolling volatility time series for sample stocks
    2. Volatility distribution comparison
    3. Index vs individual stock volatility comparison
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns
        sample_stocks (List[str]): List of stock symbols to analyze
        equal_weight_index (pd.Series): Equal-weighted index returns
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size for the plot
    
    Example:
        # >>> daily_returns = calculate_daily_returns(data)
        # >>> ew_index = build_equal_weight_index(daily_returns)
        # >>> sample_stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
        # >>> plot_volatility_analysis(daily_returns, sample_stocks, ew_index)
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Validate sample stocks and calculate volatility:
    #    - Use the implemented calculate_rolling_volatility() function
    #    - Calculate 20-day rolling volatility for both individual stocks and index
    # 2. Create 2×2 subplot layout:
    #    - Top-left: Volatility time series line plot
    #    - Top-right: Volatility distribution box plot
    #    - Bottom-left: Risk-return scatter plot
    #    - Bottom-right: Volatility clustering analysis (squared returns)
    # 3. Draw each subplot:
    #    - ax.plot() for time series
    #    - ax.boxplot() for distribution
    #    - ax.scatter() for scatter plot
    # 4. Add legends, labels, grids, etc.
    # 5. Save the plot (if path is specified)
    #
    # No return value, directly display the chart

    # Filter sample_stocks to only include those present in daily_returns
    valid_stocks = [s for s in sample_stocks if s in daily_returns.columns]
    if not valid_stocks:
        raise ValueError("None of the specified sample stocks are in the daily_returns data.")

    #Step 1: Validate sample stocks and calculate rolling volatility
    stock_vol = calculate_rolling_volatility(daily_returns[valid_stocks], window=20)
    index_vol = calculate_rolling_volatility(equal_weight_index.to_frame("Index"), window=20)

    # Step 2: Create subplot layout (2x2)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Step 3a: Top-left - Volatility time series line plot
    for stock in valid_stocks:
        ax1.plot(stock_vol.index, stock_vol[stock], label=stock)
    ax1.plot(index_vol.index, index_vol["Index"], label="Equal-Weight Index", color="black", linestyle="--")
    ax1.set_title("20-Day Rolling Volatility (Time Series)")
    ax1.set_ylabel("Volatility")
    ax1.legend()
    ax1.grid(True)

    # Step 3b: Top-right - Volatility distribution box plot
    ax2.boxplot([stock_vol[s].dropna() for s in valid_stocks], labels=valid_stocks)
    ax2.set_title("Volatility Distribution Comparison")
    ax2.set_ylabel("Volatility")
    ax2.grid(True)

    # Step 3c: Bottom-left - Risk-return scatter plot
    avg_returns = daily_returns[valid_stocks].mean() * 252
    volatilities = daily_returns[valid_stocks].std() * np.sqrt(252)
    ax3.scatter(volatilities, avg_returns, color="blue")
    for stock in valid_stocks:
        ax3.text(volatilities[stock], avg_returns[stock], stock)
    ax3.set_title("Risk-Return Scatter Plot")
    ax3.set_xlabel("Annualized Volatility")
    ax3.set_ylabel("Average Return")
    ax3.grid(True)

    # Step 3d: Bottom-right - Volatility clustering (squared returns)
    for stock in valid_stocks:
        ax4.plot(daily_returns.index, daily_returns[stock] ** 2, label=stock)
    ax4.set_title("Volatility Clustering (Squared Returns)")
    ax4.set_ylabel("Squared Return")
    ax4.legend()
    ax4.grid(True)

    # Step 4: Adjust layout and save/show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def calculate_volatility_statistics(
    daily_returns: pd.DataFrame,
    equal_weight_index: pd.Series,
    window: int = 20
) -> Dict[str, Any]:
    """
    Calculate comprehensive volatility statistics for stocks and index.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns
        equal_weight_index (pd.Series): Equal-weighted index returns
        window (int): Rolling window for volatility calculation
    
    Returns:
        Dict[str, Any]: Dictionary containing volatility statistics
    
    Example:
        # >>> stats = calculate_volatility_statistics(daily_returns, ew_index)
        # >>> print(f"Average stock volatility: {stats['avg_stock_volatility']:.4f}")
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Compute rolling volatility for stocks and index:
    #    - Use calculate_rolling_volatility()
    #    - Convert equal_weight_index to DataFrame when needed
    # 2. Stock volatility statistics:
    #    - Average volatility: stock_vol_means.mean()
    #    - Median, min, max
    #    - Volatility of volatilities: stock_vol_means.std()
    # 3. Index volatility statistics
    # 4. Diversification benefit:
    #    - diversification_ratio = avg_stock_volatility / index_volatility
    # 5. Correlation between stock and market volatilities
    # 6. Return statistics dict
    #
    # Expected output: Dict of volatility statistics

    # Step 1: Compute rolling volatility for stocks and index
    stock_vol = calculate_rolling_volatility(daily_returns, window=window)
    index_vol = calculate_rolling_volatility(equal_weight_index.to_frame("Index"), window=window)

    # Step 2: Stock volatility statistics
    stock_vol_means = stock_vol.mean()  # average vol per stock
    avg_stock_volatility = stock_vol_means.mean()
    median_stock_volatility = stock_vol_means.median()
    min_stock_volatility = stock_vol_means.min()
    max_stock_volatility = stock_vol_means.max()
    volatility_of_vols = stock_vol_means.std()

    # Step 3: Index volatility statistics
    avg_index_volatility = index_vol["Index"].mean()
    max_index_volatility = index_vol["Index"].max()
    min_index_volatility = index_vol["Index"].min()
    index_vol_std = index_vol["Index"].std()

    # Step 4: Diversification benefit
    diversification_ratio = avg_stock_volatility / avg_index_volatility if avg_index_volatility > 0 else np.nan

    # Step 5: Correlation between stock volatilities and index volatility
    correlation_with_index = stock_vol.corrwith(index_vol["Index"])

    # Step 6: Return results in dictionary
    stats = {
        "avg_stock_volatility": avg_stock_volatility,
        "median_stock_volatility": median_stock_volatility,
        "min_stock_volatility": min_stock_volatility,
        "max_stock_volatility": max_stock_volatility,
        "vol_of_volatilities": volatility_of_vols,
        "index_volatility": avg_index_volatility,
        "index_vol_std": index_vol_std,
        "min_index_volatility": min_index_volatility,
        "max_index_volatility": max_index_volatility,
        "diversification_ratio": diversification_ratio,
        "n_stocks_analyzed": len(stock_vol.columns),
        "correlation_with_index": correlation_with_index
    }

    return stats

def compare_individual_vs_market(
    daily_returns: pd.DataFrame,
    equal_weight_index: pd.Series
) -> pd.DataFrame:
    """
    Compare individual stock performance with market index.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns
        equal_weight_index (pd.Series): Equal-weighted index returns
    
    Returns:
        pd.DataFrame: Comparison statistics for each stock
    
    Example:
        # >>> comparison = compare_individual_vs_market(daily_returns, ew_index)
        # >>> print(comparison.head())
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. For each stock vs market index:
    #    - Align data: reindex() to ensure time alignment
    #    - Filter insufficient data (<50 observations)
    # 2. Compute key metrics:
    #    - Annualized return and volatility (×252, ×√252)
    #    - Beta: covariance / market variance
    #    - Alpha: stock return - Beta × market return
    #    - Correlation: stock.corr(index)
    # 3. Risk-adjusted metrics:
    #    - Sharpe ratio: annualized return / annualized volatility
    #    - Tracking error: std of (stock - Beta × market)
    #    - Information ratio: alpha / tracking error
    # 4. Organize results as DataFrame
    #
    # Expected output: DataFrame, rows=stocks, columns=metrics
    if daily_returns is None or equal_weight_index is None:
        raise ValueError("Input data cannot be None.")

    if daily_returns.empty or equal_weight_index.empty:
        return pd.DataFrame(columns=[
            "annualized_return", "annualized_volatility", "beta", "alpha",
            "correlation", "sharpe_ratio", "tracking_error", "information_ratio", "observations"
        ])

    results = {}  # Store the indicators of each stock in a dictionary

    # Step 1: Traverse each stock and compare it with the market index one by one
    for stock in daily_returns.columns:
        # Align the time index of stocks and markets
        aligned = pd.concat([daily_returns[stock], equal_weight_index], axis=1, join="inner")
        aligned.columns = ["stock", "market"]

        # If there are less than 50 valid data, skip
        valid_data = aligned.dropna()
        observations = len(valid_data)
        if observations < 50:
            continue

        # Step 2: Calculate annualized returns and volatility
        ann_return_stock = valid_data["stock"].mean() * 252
        ann_vol_stock = valid_data["stock"].std() * np.sqrt(252)
        ann_return_market = valid_data["market"].mean() * 252
        ann_vol_market = valid_data["market"].std() * np.sqrt(252)

        # Step 3: Calculate Beta and Alpha
        cov = valid_data.cov().iloc[0, 1]  # 股票和市场的协方差
        var_market = valid_data["market"].var()
        beta = cov / var_market if var_market != 0 else np.nan
        alpha = ann_return_stock - beta * ann_return_market if np.isfinite(beta) else np.nan

        # Step 4: Calculate correlation coefficient
        corr = aligned["stock"].corr(aligned["market"])

        # Step 5: Risk-adjusted metrics
        sharpe = ann_return_stock / ann_vol_stock if ann_vol_stock != 0 else np.nan
        # tracking error: std of (stock - beta * market)
        if np.isfinite(beta):
            tracking_error = (valid_data["stock"] - beta * valid_data["market"]).std() * np.sqrt(252)
        else:
            tracking_error = np.nan

        info_ratio = alpha / tracking_error if tracking_error != 0 else np.nan

        # Step 6: Store the results in a dictionary
        results[stock] = {
            "annualized_return": ann_return_stock,
            "annualized_volatility": ann_vol_stock,
            "beta": beta,
            "alpha": alpha,
            "correlation": corr,
            "sharpe_ratio": sharpe,
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "observations": observations
        }

    # Ensure deterministic column order when transposed
    df = pd.DataFrame(results).T
    # If empty, return with expected columns
    if df.empty:
        return pd.DataFrame(columns=[
            "annualized_return", "annualized_volatility", "beta", "alpha",
            "correlation", "sharpe_ratio", "tracking_error", "information_ratio", "observations"
        ])
    # Reorder columns to a stable order
    cols = ["annualized_return", "annualized_volatility", "beta", "alpha",
            "correlation", "sharpe_ratio", "tracking_error", "information_ratio", "observations"]
    # keep only existing + in the requested order
    cols_existing = [c for c in cols if c in df.columns]
    return df[cols_existing]



# Example usage and testing functions
def main():

    #test code
    # np.random.seed(42)
    # dates = pd.date_range(start="2025-01-01", periods=100, freq="B")
    # stocks = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D", "STOCK_E"]
    # daily_returns = pd.DataFrame(np.random.normal(0.0005, 0.02, size=(100, 5)),
    #                              index=dates, columns=stocks)
    #
    # ew_index = build_equal_weight_index(daily_returns)
    # volatility = calculate_rolling_volatility(daily_returns)
    # stats = calculate_volatility_statistics(daily_returns, ew_index)
    # comparison = compare_individual_vs_market(daily_returns, ew_index)
    #
    # sample_stocks = ["STOCK_A", "STOCK_B", "STOCK_C"]
    # plot_volatility_analysis(daily_returns, sample_stocks, ew_index)
    #
    # print("Volatility statistics:\n", stats)
    # print("Individual vs Market comparison:\n", comparison.head())


    """
    Main function to demonstrate the usage of volatility analysis functions.
    
    This function provides examples of how to use the implemented functions
    and can be used for testing purposes.
    """
    print("Task 2: Market & Asset Characterization")
    print("=" * 50)
    
    print("Functions implemented:")
    print("1. calculate_rolling_volatility() - Calculate rolling volatility for stocks")
    print("2. build_equal_weight_index() - Create equal-weighted market index")
    print("3. plot_volatility_analysis() - Comprehensive volatility visualization")
    print("4. calculate_volatility_statistics() - Volatility statistics summary")
    print("5. compare_individual_vs_market() - Individual vs market comparison")
    
    print("\nExample usage:")
    print("""
    # Calculate daily returns first
    # daily_returns = daily_data.pct_change().dropna()
    
    # Calculate rolling volatility
    # volatility = calculate_rolling_volatility(daily_returns, window=20)
    
    # Build equal-weight index
    # ew_index = build_equal_weight_index(daily_returns)
    
    # Create volatility analysis plots
    # sample_stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
    # plot_volatility_analysis(daily_returns, sample_stocks, ew_index)
    
    # Calculate statistics
    # stats = calculate_volatility_statistics(daily_returns, ew_index)
    # comparison = compare_individual_vs_market(daily_returns, ew_index)
    """)


if __name__ == "__main__":
    main()


