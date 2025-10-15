
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用无界面后端，避免 “没有名称为 'pyplot' 的模块” 报错
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')


def calculate_correlation_matrix(
        daily_returns: pd.DataFrame,
        method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix of daily returns for all stocks.

    This function computes the pairwise correlation between all stocks
    in the dataset over the entire period. The correlation matrix shows
    how stock returns move together.

    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns for each stock
                                    Columns: stock symbols
                                    Index: dates
        method (str): Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        pd.DataFrame: Correlation matrix with stocks as both rows and columns
                     Values range from -1 (perfect negative correlation) to
                     +1 (perfect positive correlation)

    Example:
        >>> # Self-contained example with random data (no external dependencies)
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range('2020-01-01', periods=100)
        >>> daily_returns = pd.DataFrame(
        ...     np.random.randn(100, 4) / 100.0,
        ...     index=idx, columns=['A','B','C','D']
        ... )
        >>> corr_matrix = calculate_correlation_matrix(daily_returns)
        >>> isinstance(corr_matrix, pd.DataFrame)
        True

    Notes:
        - Diagonal elements are always 1.0 (perfect self-correlation)
        - Matrix is symmetric
        - Missing data is handled by pairwise deletion
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Use pandas corr() function to calculate correlation matrix:
    #    - daily_returns.corr(method=method)
    # 2. method parameter supports: 'pearson', 'spearman', 'kendall'
    # 3. Function automatically handles missing values (pairwise deletion)
    # 4. Returned matrix should be symmetric with diagonal = 1.0
    #
    # Expected output: DataFrame with stock codes as rows and columns, values as correlation coefficients

    # Validate input
    if not isinstance(daily_returns, pd.DataFrame):
        raise TypeError("daily_returns must be a pandas DataFrame")
    if method not in ('pearson', 'spearman', 'kendall'):
        raise ValueError("method must be one of: 'pearson', 'spearman', 'kendall'")
    if daily_returns.shape[1] == 0:
        return pd.DataFrame()

    # Calculate correlation matrix using pairwise deletion by default
    corr_matrix = daily_returns.corr(method=method)

    # Ensure symmetry and proper diagonal values (numerical stability)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2.0
    np.fill_diagonal(corr_matrix.values, 1.0)

    return corr_matrix


def calculate_rolling_correlation(
        stock1_returns: pd.Series,
        stock2_returns: pd.Series,
        window: int = 60,
        min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling correlation between two specific stocks.

    This function computes the correlation between two stocks over a rolling
    window to analyze how their relationship changes over time.

    Args:
        stock1_returns (pd.Series): Returns for first stock
        stock2_returns (pd.Series): Returns for second stock
        window (int): Rolling window size in days (default 60 for ~3 months)
        min_periods (Optional[int]): Minimum periods required for calculation

    Returns:
        pd.Series: Rolling correlation time series
                  Index: dates
                  Values: correlation coefficients

    Example:
        >>> # Self-contained example
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range('2020-01-01', periods=120)
        >>> df = pd.DataFrame(
        ...     np.random.randn(120, 2)/100,
        ...     index=idx, columns=['X','Y']
        ... )
        >>> rc = calculate_rolling_correlation(df['X'], df['Y'], window=30)
        >>> isinstance(rc, pd.Series)
        True

    Notes:
        - The first (window-1) observations will be NaN
        - Correlation can vary significantly over time
        - Values range from -1 to +1
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Align two return series:
    #    - Create DataFrame with two columns
    #    - Use dropna() to remove missing values
    # 2. Calculate rolling correlation:
    #    - Use stock1.rolling(window).corr(stock2)
    #    - Set appropriate min_periods parameter
    # 3. Handle parameters:
    #    - If min_periods is None, set to window
    # 4. Return time series
    #
    # Expected output: Series with time index and rolling correlation coefficients as values

    if not isinstance(stock1_returns, pd.Series) or not isinstance(stock2_returns, pd.Series):
        raise TypeError("stock1_returns and stock2_returns must be pandas Series")

    # Align on common index and drop NaNs pairwise
    df = pd.concat([stock1_returns, stock2_returns], axis=1, keys=['s1', 's2']).dropna()

    if df.empty:
        return pd.Series(dtype=float)

    if min_periods is None:
        min_periods = window

    # Rolling correlation
    rolling_corr = df['s1'].rolling(window=window, min_periods=min_periods).corr(df['s2'])

    return rolling_corr


def plot_correlation_heatmap(
        correlation_matrix: pd.DataFrame,
        title: str = "Stock Returns Correlation Matrix",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = 'coolwarm'
) -> None:
    """
    Create a heatmap visualization of the correlation matrix.

    This function generates a color-coded heatmap to visualize the correlation
    structure between all stocks in the dataset.

    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix to visualize
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size
        cmap (str): Colormap for the heatmap

    Example:
        >>> # Self-contained example
        >>> import pandas as pd, numpy as np
        >>> cm = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], columns=['A','B'], index=['A','B'])
        >>> plot_correlation_heatmap(cm, title="Demo Heatmap")  # doctest: +ELLIPSIS
        >>> # A figure will be shown; function returns None.

    Notes:
        - Red colors indicate positive correlation
        - Blue colors indicate negative correlation
        - Diagonal is always dark red (correlation = 1)
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Create figure: plt.figure(figsize=figsize)
    # 2. Optional: create upper-triangular mask to avoid duplicate display
    #    - mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # 3. Plot heatmap with seaborn:
    #    - sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, ...)
    #    - set center=0 for neutral zero
    # 4. Add title and labels:
    #    - plt.title(), plt.xlabel(), plt.ylabel()
    #    - rotate x-axis labels for readability
    # 5. Save and show figure
    #
    # No return value; show heatmap directly

    if not isinstance(correlation_matrix, pd.DataFrame):
        raise TypeError("correlation_matrix must be a pandas DataFrame")
    if correlation_matrix.empty:
        warnings.warn("Empty correlation matrix provided; nothing to plot.")
        return

    plt.figure(figsize=figsize)

    # Upper triangular mask to avoid duplicate info
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        annot=False,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
    )
    plt.title(title)
    plt.xlabel("Stocks")
    plt.ylabel("Stocks")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_rolling_correlation_analysis(
        daily_returns: pd.DataFrame,
        stock_pairs: List[Tuple[str, str]],
        window: int = 60,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
) -> Dict[Tuple[str, str], pd.Series]:
    """
    Analyze and visualize rolling correlations for multiple stock pairs.

    This function calculates and plots rolling correlations for specified
    stock pairs to show how relationships change over time.

    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns
        stock_pairs (List[Tuple[str, str]]): List of stock pairs to analyze
        window (int): Rolling window size
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size

    Returns:
        Dict[Tuple[str, str], pd.Series]: Dictionary of rolling correlation series

    Example:
        >>> # Self-contained example
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range('2020-01-01', periods=200)
        >>> dr = pd.DataFrame(
        ...     np.random.randn(200, 3)/100,
        ...     index=idx, columns=['S1','S2','S3']
        ... )
        >>> res = plot_rolling_correlation_analysis(dr, [('S1','S2'), ('S2','S3')], window=30)  # doctest: +ELLIPSIS
        >>> isinstance(res, dict)
        True
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Validate stock pairs exist in data
    # 2. Compute rolling correlation for each pair:
    #    - Use implemented calculate_rolling_correlation()
    # 3. Create subplot layout:
    #    - Determine n_rows, n_cols
    #    - Use plt.subplots() to create subplots
    # 4. Plot each pair's rolling correlation:
    #    - ax.plot() time series
    #    - Add zero line and mean line as references
    #    - Set y-limits to (-1.1, 1.1)
    # 5. Add titles, labels, legend, grid
    # 6. Hide extra subplots
    # 7. Save and show figure
    #
    # Expected output: Dict[Tuple[pair], Series[rolling_correlation]]

    if not isinstance(daily_returns, pd.DataFrame):
        raise TypeError("daily_returns must be a pandas DataFrame")
    if not stock_pairs:
        warnings.warn("No stock pairs provided; nothing to analyze.")
        return {}

    # Validate pairs
    available = set(daily_returns.columns)
    valid_pairs: List[Tuple[str, str]] = []
    invalid_pairs: List[Tuple[str, str]] = []
    for a, b in stock_pairs:
        if a in available and b in available:
            valid_pairs.append((a, b))
        else:
            invalid_pairs.append((a, b))

    if invalid_pairs:
        # 为配合你贴的 tests：提供严格错误以便被捕获
        raise ValueError(f"Invalid pairs: {invalid_pairs}")

    if not valid_pairs:
        warnings.warn("No valid stock pairs after validation.")
        return {}

    n = len(valid_pairs)
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    results: Dict[Tuple[str, str], pd.Series] = {}

    for idx, (a, b) in enumerate(valid_pairs):
        ax = axes[idx]
        rc = calculate_rolling_correlation(daily_returns[a], daily_returns[b], window=window)
        results[(a, b)] = rc

        ax.plot(rc.index, rc.values, label=f"{a} vs {b}", color='tab:blue', lw=1.5)
        ax.axhline(0.0, color='gray', ls='--', lw=1)
        if not rc.dropna().empty:
            mean_val = rc.mean()
            ax.axhline(mean_val, color='tab:red', ls=':', lw=1, label=f"Mean={mean_val:.2f}")

        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"Rolling Correlation ({a} vs {b}) - window={window}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Correlation")
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)

    # Hide extra subplots
    for j in range(len(valid_pairs), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    return results


def analyze_correlation_structure(
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze the correlation structure of the stock universe.

    This function provides insights into the correlation patterns
    among stocks, including clustering and distribution analysis.

    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix to analyze
        threshold (float): Correlation threshold for high correlation pairs

    Returns:
        Dict[str, Any]: Dictionary containing correlation analysis results

    Example:
        >>> # Self-contained example
        >>> import pandas as pd, numpy as np
        >>> cm = pd.DataFrame(
        ...     [[1.0, 0.6, -0.2],
        ...      [0.6, 1.0,  0.1],
        ...      [-0.2, 0.1, 1.0]],
        ...     columns=['A','B','C'], index=['A','B','C']
        ... )
        >>> analysis = analyze_correlation_structure(cm)
        >>> 'avg_correlation' in analysis and 'high_corr_pairs' in analysis
        True
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Extract off-diagonal correlation values:
    #    - Use np.eye() to create diagonal mask
    #    - Extract off-diagonal elements
    # 2. Compute basic statistics:
    #    - mean, median, std, min, max
    # 3. Identify high-correlation pairs:
    #    - Iterate upper triangle
    #    - Find pairs with abs(corr) >= threshold
    # 4. Distribution characteristics:
    #    - Positive vs negative proportions
    #    - Count and ratio of high-corr pairs
    # 5. Quantiles
    # 6. Return result dictionary
    #
    # Expected output: Dict with correlation structure metrics

    if not isinstance(correlation_matrix, pd.DataFrame):
        raise TypeError("correlation_matrix must be a pandas DataFrame")
    if correlation_matrix.empty:
        return {
            'avg_correlation': np.nan,
            'median_correlation': np.nan,
            'std_correlation': np.nan,
            'min_correlation': np.nan,
            'max_correlation': np.nan,
            'quantiles': {},
            'n_pairs': 0,
            'n_positive_corr': 0,
            'n_negative_corr': 0,
            'pct_positive_corr': np.nan,
            'pct_negative_corr': np.nan,
            'high_corr_pairs': [],
            'n_high_corr_pairs': 0,
            'pct_high_corr_pairs': np.nan,
            # ADDED for tests compatibility
            'n_stocks': 0,
            'pct_high_corr': np.nan,
            'q25': np.nan,
            'q75': np.nan,
        }

    mat = correlation_matrix.values.astype(float)
    n = mat.shape[0]
    cols = list(correlation_matrix.columns)

    # Off-diagonal values
    if n <= 1:
        off_diag_vals = np.array([])
    else:
        mask_offdiag = ~np.eye(n, dtype=bool)
        off_diag_vals = mat[mask_offdiag]

    # Unique pairs (upper triangle)
    iu = np.triu_indices(n, k=1)
    upper_vals = mat[iu]

    # Basic stats
    mean_val = np.nanmean(off_diag_vals) if off_diag_vals.size > 0 else np.nan
    median_val = np.nanmedian(off_diag_vals) if off_diag_vals.size > 0 else np.nan
    std_val = np.nanstd(off_diag_vals, ddof=0) if off_diag_vals.size > 0 else np.nan
    min_val = np.nanmin(off_diag_vals) if off_diag_vals.size > 0 else np.nan
    max_val = np.nanmax(off_diag_vals) if off_diag_vals.size > 0 else np.nan

    n_pairs = upper_vals.size
    n_pos = int(np.nansum(upper_vals > 0))
    n_neg = int(np.nansum(upper_vals < 0))
    pct_pos = (n_pos / n_pairs * 100.0) if n_pairs > 0 else np.nan
    pct_neg = (n_neg / n_pairs * 100.0) if n_pairs > 0 else np.nan

    # High-correlation pairs (return dicts to satisfy tests)
    high_pairs_dicts: List[Dict[str, Any]] = []
    for i, j, v in zip(iu[0], iu[1], upper_vals):
        if np.isfinite(v) and abs(v) >= threshold:
            high_pairs_dicts.append({'stock1': cols[i], 'stock2': cols[j], 'correlation': float(v)})
    n_high = len(high_pairs_dicts)
    pct_high = (n_high / n_pairs * 100.0) if n_pairs > 0 else np.nan

    # Quantiles
    quantiles = {}
    q25 = q75 = np.nan
    if off_diag_vals.size > 0:
        qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        q_vals = np.nanpercentile(off_diag_vals, [q * 100 for q in qs])
        quantiles = {f"q{int(q * 100)}": float(val) for q, val in zip(qs, q_vals)}
        q25 = float(np.nanpercentile(off_diag_vals, 25))
        q75 = float(np.nanpercentile(off_diag_vals, 75))

    # Build result dict (keep previous keys + tests-required keys)
    result = {
        'avg_correlation': float(mean_val) if np.isfinite(mean_val) else np.nan,
        'median_correlation': float(median_val) if np.isfinite(median_val) else np.nan,
        'std_correlation': float(std_val) if np.isfinite(std_val) else np.nan,
        'min_correlation': float(min_val) if np.isfinite(min_val) else np.nan,
        'max_correlation': float(max_val) if np.isfinite(max_val) else np.nan,
        'quantiles': quantiles,
        'n_pairs': int(n_pairs),
        'n_positive_corr': int(n_pos),
        'n_negative_corr': int(n_neg),
        'pct_positive_corr': float(pct_pos) if np.isfinite(pct_pos) else np.nan,
        'pct_negative_corr': float(pct_neg) if np.isfinite(pct_neg) else np.nan,
        'high_corr_pairs': high_pairs_dicts,  # ADJUSTED: list of dicts
        'n_high_corr_pairs': int(n_high),
        'pct_high_corr_pairs': float(pct_high) if np.isfinite(pct_high) else np.nan,
        # ADDED for tests compatibility
        'n_stocks': int(n),
        'pct_high_corr': float(pct_high) if np.isfinite(pct_high) else np.nan,
        'q25': q25,
        'q75': q75,
    }
    return result


def plot_correlation_distribution(
        correlation_matrix: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the distribution of pairwise correlations.

    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix to analyze
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Extract off-diagonal correlation values:
    #    - Use diagonal mask to drop diagonal entries
    # 2. Create 1x2 subplot layout:
    #    - Left: histogram of correlation distribution
    #    - Right: boxplot summary
    # 3. Plot histogram:
    #    - plt.hist()
    #    - Add mean/median reference lines
    #    - Add legend and labels
    # 4. Plot boxplot:
    #    - plt.boxplot()
    # 5. Add stats textbox: count, mean, std, min, max
    # 6. Save and show
    #
    # No return value; show the distribution plots

    if not isinstance(correlation_matrix, pd.DataFrame):
        raise TypeError("correlation_matrix must be a pandas DataFrame")
    if correlation_matrix.empty:
        warnings.warn("Empty correlation matrix provided; nothing to plot.")
        return

    mat = correlation_matrix.values.astype(float)
    n = mat.shape[0]
    if n <= 1:
        warnings.warn("Correlation matrix too small for distribution plot.")
        return

    # Extract off-diagonal values (unique pairs - upper triangle)
    iu = np.triu_indices(n, k=1)
    vals = mat[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        warnings.warn("No finite correlation values to plot.")
        return

    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    std_val = float(np.std(vals, ddof=0))
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    count = int(vals.size)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax0 = axes[0]
    ax0.hist(vals, bins=30, color='tab:blue', alpha=0.75, edgecolor='white')
    ax0.axvline(mean_val, color='tab:red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    ax0.axvline(median_val, color='tab:green', linestyle=':', linewidth=1.5, label=f"Median: {median_val:.2f}")
    ax0.set_title("Correlation Distribution (Histogram)")
    ax0.set_xlabel("Correlation")
    ax0.set_ylabel("Frequency")
    ax0.legend()
    ax0.grid(True, ls='--', alpha=0.4)

    # Boxplot
    ax1 = axes[1]
    ax1.boxplot(vals, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgray', color='black'),
                medianprops=dict(color='tab:red', linewidth=2))
    ax1.set_title("Correlation Summary (Boxplot)")
    ax1.set_ylabel("Correlation")
    ax1.set_xticks([1])
    ax1.set_xticklabels(["All Pairs"])

    # Stats textbox
    stats_text = (
        f"Count: {count}\n"
        f"Mean: {mean_val:.3f}\n"
        f"Std: {std_val:.3f}\n"
        f"Min: {min_val:.3f}\n"
        f"Max: {max_val:.3f}"
    )
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


# ====== Optional helper functions for 5-minute data -> daily returns (added, no impact on your tasks) ======
import os


def load_5min_data(path: str,
                   symbol_col: str = 'symbol',
                   datetime_col: str = 'datetime',
                   price_col: str = 'close',
                   tz: Optional[str] = None) -> pd.DataFrame:
    """
    Load 5-minute bar data and return a wide price table.
    - path: CSV/Parquet 文件或包含多文件的目录
    - 返回: index=时间的 DatetimeIndex，columns=股票代码，values=收盘价
    """

    def read_one(fp: str) -> pd.DataFrame:
        if fp.lower().endswith('.parquet'):
            df = pd.read_parquet(fp)
        else:
            try:
                df = pd.read_csv(fp)
            except UnicodeDecodeError:
                df = pd.read_csv(fp, encoding='utf-8-sig')
        # 列名大小写兼容
        cols = {c.lower(): c for c in df.columns}
        sc = cols.get(symbol_col.lower(), symbol_col)
        dc = cols.get(datetime_col.lower(), datetime_col)
        pc = cols.get(price_col.lower(), price_col)
        # 若缺少 symbol 列，用文件名作为 symbol
        if sc not in df.columns:
            sym = os.path.splitext(os.path.basename(fp))[0]
            df['sym'] = sym
            sc = 'sym'
        # 解析时间列
        df[dc] = pd.to_datetime(df[dc], errors='coerce', utc=False)
        if tz:
            df[dc] = df[dc].dt.tz_localize(tz, nonexistent='shift_forward', ambiguous='NaT')
        df = df.dropna(subset=[dc, pc])
        df = df[[dc, sc, pc]].rename(columns={dc: 'dt', sc: 'sym', pc: 'px'})
        return df

    files = []
    if os.path.isdir(path):
        for fn in os.listdir(path):
            if fn.lower().endswith(('.csv', '.parquet')):
                files.append(os.path.join(path, fn))
    else:
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet files found under: {path}")

    parts = [read_one(fp) for fp in files]
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.drop_duplicates(subset=['dt', 'sym']).sort_values(['dt', 'sym'])
    wide = raw.pivot(index='dt', columns='sym', values='px').sort_index()
    return wide


def resample_to_daily_returns(wide_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-minute prices to daily close and compute daily returns.
    """
    if not isinstance(wide_5m.index, pd.DatetimeIndex):
        raise TypeError("Input must have a DatetimeIndex")
    daily_close = wide_5m.resample('1D').last()
    daily_returns = daily_close.pct_change().dropna(how='all')
    return daily_returns


# ====== End of optional helper functions ======


# Example usage and testing functions
def main():
    """
    Main function to demonstrate the usage of correlation analysis functions.

    This function provides examples of how to use the implemented functions
    and can be used for testing purposes.
    """
    print("Task 3: Cross-Sectional Analysis")
    print("=" * 50)

    print("Functions implemented:")
    print("1. calculate_correlation_matrix() - Calculate correlation matrix")
    print("2. calculate_rolling_correlation() - Rolling correlation for stock pairs")
    print("3. plot_correlation_heatmap() - Visualize correlation matrix")
    print("4. plot_rolling_correlation_analysis() - Rolling correlation analysis")
    print("5. analyze_correlation_structure() - Correlation structure analysis")
    print("6. plot_correlation_distribution() - Correlation distribution plots")

    print("\nExample usage:")
    print("""
    # Calculate daily returns first
    # daily_returns = daily_data.pct_change().dropna()

    # Calculate correlation matrix
    # corr_matrix = calculate_correlation_matrix(daily_returns)

    # Visualize correlation matrix
    # plot_correlation_heatmap(corr_matrix)

    # Analyze rolling correlations
    # stock_pairs = [('STOCK_1', 'STOCK_2'), ('STOCK_3', 'STOCK_4')]
    # rolling_corrs = plot_rolling_correlation_analysis(daily_returns, stock_pairs)

    # Analyze correlation structure
    # analysis = analyze_correlation_structure(corr_matrix)
    # plot_correlation_distribution(corr_matrix)
    """)

    # ADDED: minimal demo to ensure visible outputs and files
    try:
        print("\n[Demo] Generating synthetic daily returns...")
        idx = pd.date_range('2020-01-01', periods=120, freq='B')
        daily_returns = pd.DataFrame(
            np.random.randn(len(idx), 4) / 100.0,
            index=idx, columns=['A', 'B', 'C', 'D']
        )
        print("[Demo] daily_returns shape:", daily_returns.shape)

        corr = calculate_correlation_matrix(daily_returns)
        print("[Demo] Correlation matrix shape:", corr.shape)
        avg_off = float(np.nanmean(corr.values[np.triu_indices_from(corr, 1)]))
        print(f"[Demo] Avg off-diagonal correlation: {avg_off:.4f}")

        plot_correlation_heatmap(corr, save_path='corr_heatmap.png')
        plot_correlation_distribution(corr, save_path='corr_dist.png')
        _ = plot_rolling_correlation_analysis(
            daily_returns,
            stock_pairs=[('A', 'B'), ('C', 'D')],
            window=30,
            save_path='rolling_corr.png'
        )
        analysis = analyze_correlation_structure(corr, threshold=0.5)
        # 打印单测里关心的若干字段
        print("[Demo] Structure:", {
            'avg_correlation': analysis['avg_correlation'],
            'n_stocks': analysis['n_stocks'],
            'n_pairs': analysis['n_pairs'],
            'n_high_corr_pairs': analysis['n_high_corr_pairs'],
            'pct_high_corr': analysis['pct_high_corr']
        })
        print("[Demo] Saved files: corr_heatmap.png, corr_dist.png, rolling_corr.png")
    except Exception as e:
        print("[Demo] Failed:", e)


if __name__ == "__main__":
    main()
