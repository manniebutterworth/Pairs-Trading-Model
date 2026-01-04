print("--------------------------------------------------------------------")
print("---------- STATISTICAL ARBITRAGE PAIRS TRADING MODEL v1.4 ----------")
print("--------------------------------------------------------------------")

# ---------- IMPORT MODULES ----------

print("Loading Modules...")

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import statsmodels.api as sm

# ---------- CONTROLS ---------- 

training_data_cutoff = 0.7
data_relevance_period = 90
reversion_frequency_threshold = 0.2
regularisation_lambda = 10
deviation_range_intervals = 100
number_of_pairs_selected_from_backtest = 5
reversion_time_multiplier = 2
rf = 0.03

# ------------------------------------------
print("---------- PHASE 1: DATA ----------")
# ------------------------------------------

# ---------- LOAD PRICE DATA ----------

price_data = np.loadtxt("price_data.txt")
print("Price Data Shape:", price_data.shape)

def training_split(data):
    """
    Return the training slice from the start of the data.
    """
    training_data = data[:data_relevance_period]
    return training_data

def test_split(data):
    """
    Return the test slice of the data based on the training cutoff.
    """
    data_rows = int(data.shape[0])
    training_data_rows = int(training_data_cutoff * data_rows)
    test_data = data[training_data_rows:data_rows]
    return test_data

training_price_data = training_split(price_data)
test_price_data = test_split(price_data)
print("Training Price Data Shape:", training_price_data.shape)
print("Test Price Data Shape:", test_price_data.shape)

# ---------- CALCULATE LOG DATA ----------

log_price_data = np.log(price_data)
training_log_price_data = training_split(log_price_data)
test_log_price_data = test_split(log_price_data)
print("Training Log Price Data Shape:", training_log_price_data.shape)
print("Test Log Price Data Shape:", test_log_price_data.shape)

# ---------- CALCULATE RETURN DATA ----------

return_data = np.diff(log_price_data, axis=0)
training_return_data = training_split(return_data)
test_return_data = test_split(return_data)
print("Training Return Data Shape:", training_return_data.shape)
print("Test Return Data Shape:", test_return_data.shape)

# ---------- CLASSIFY RELEVANT TRAINING DATA ----------

relevant_training_price_data = training_price_data[-data_relevance_period:]
relevant_training_log_price_data = training_log_price_data[-data_relevance_period:]
relevant_training_return_data = training_return_data[-data_relevance_period:]

# ------------------------------------------------------------
print("---------- PHASE 2: COINTEGRATION ANALYSIS ----------")
# ------------------------------------------------------------

# ---------- CALCULATE UNIQUE PAIRS ----------

asset_count = price_data.shape[1]
pairs = np.array(list(combinations(range(asset_count), 2)))
print("Number of Assets:", asset_count)
print("Unique Pairs Shape:", pairs.shape)

# ---------- CALCULATE COINTEGRATION COEFFICIENTS & PAIR ORDER ----------

def pair_order_cointegration(price_data, pairs):
    """
    For each candidate pair, pick the direction that gives the stronger OLS slope.

    Runs two regressions per pair, A ~ B and B ~ A, then selects the one with the
    larger absolute slope coefficient. The chosen direction is returned as
    (independent, dependent) alongside the selected slope for each pair.

    Args:
        price_data: 2D array of prices with shape (time, assets).
        pairs: Iterable of (A, B) index pairs into the asset dimension.

    Returns:
        ordered_pairs: Array of (independent, dependent) indices for each pair.
        coefficients: Array of selected slope coefficients (one per pair).
    """
    ordered_pairs = []
    coefficients = []

    for A, B in pairs:
        price_A = price_data[:, A]
        price_B = price_data[:, B]

        # Regression 1: A ~ B
        X1 = sm.add_constant(price_B)
        model1 = sm.OLS(price_A, X1).fit()
        coeff1 = model1.params[1]
        independent1 = B
        dependent1 = A

        # Regression 2: B ~ A
        X2 = sm.add_constant(price_A)
        model2 = sm.OLS(price_B, X2).fit()
        coeff2 = model2.params[1]
        independent2 = A
        dependent2 = B

        # Choose the larger absolute coefficient
        if abs(coeff1) >= abs(coeff2):
            coeff = coeff1
            independent  = independent1
            dependent  = dependent1
        else:
            coeff = coeff2
            independent  = independent2
            dependent  = dependent2

        ordered_pairs.append((independent, dependent))
        coefficients.append(coeff)

    return np.array(ordered_pairs), np.array(coefficients)

pairs_ordered, cointegration_coefficients = pair_order_cointegration(relevant_training_log_price_data, pairs)
print("Cointegration Coefficients Shape", cointegration_coefficients.shape)
print("Ordered Pairs Shape", pairs_ordered.shape)

# ---------- CALCULATE RESIDUAL SERIES & EQ VALUES ----------

def residual(price_data, ordered_pairs, CC):
    """
    Compute pair residuals using the cointegration coefficient.

    For each (A, B) pair, returns the time series residual:
        residual = price_B - CC * price_A

    Args:
        price_data: 2D array of prices with shape (time, assets).
        ordered_pairs: Array like of (A, B) indices (independent, dependent order).
        CC: 1D array of cointegration coefficients, one per pair.

    Returns:
        2D array of residuals with shape (n_pairs, time).
    """
    A_prices = []
    B_prices = []

    for A, B in ordered_pairs:
        price_A = price_data[:, A]
        A_prices.append(price_A)

        price_B = price_data[:, B]
        B_prices.append(price_B)

    A_prices = np.array(A_prices)
    B_prices = np.array(B_prices)
    CC = CC.reshape(-1, 1)

    residual = B_prices - (CC * A_prices)
    return residual

residual_series = residual(relevant_training_log_price_data, pairs_ordered, cointegration_coefficients)
print("Residual Series Shape", residual_series.shape)

equilibrium_values = np.mean(residual_series, axis=1)
print("Equilibrium Shape:", equilibrium_values.shape)

# ---------- CALCULATE REVERSION FREQUENCIES & TIMES ----------

def reversion_frequency(residual,EQ):
    """
    Estimate how often each residual series crosses its equilibrium level.

    Centers residuals around EQ, counts sign changes across time, then converts
    crossings into a per step crossing frequency and an implied average
    reversion time (in time steps).

    Args:
        residual: 2D array of residuals with shape (n_pairs, time).
        EQ: 1D array of equilibrium values, one per pair.

    Returns:
        frequency: 1D array of crossing frequency per pair.
        average_reversion_time: 1D array of average time between crossings.
    """
    EQ_matrix = EQ[:, np.newaxis]
    centered = residual - EQ_matrix 

    sign_changes = np.diff(np.sign(centered), axis=1)
    crossings_per_series = np.sum(sign_changes != 0, axis=1)
    time_length = residual.shape[1]

    frequency = crossings_per_series / (time_length - 1)
    average_reversion_time = time_length / crossings_per_series

    return frequency, average_reversion_time

reversion_frequencies, reversion_times = reversion_frequency(residual_series,equilibrium_values)
print("Reversion Frequency Shape:", reversion_frequencies.shape)

# ---------- DEFINE TRADABLE PAIRS ----------

reversion_frequency_mask = reversion_frequencies > reversion_frequency_threshold
positive_cointegration_coefficient_mask = cointegration_coefficients > 0

reversion_and_cointegration_mask = reversion_frequency_mask & positive_cointegration_coefficient_mask
tradable_pair_index = np.where(reversion_and_cointegration_mask)[0]

tradable_pairs = pairs_ordered[tradable_pair_index]
tradable_residual_series = residual_series[tradable_pair_index]
tradable_equilibrium_values = equilibrium_values[tradable_pair_index]
tradable_cointegration_coefficients = cointegration_coefficients[tradable_pair_index]
tradable_reversion_times = reversion_times[tradable_pair_index]

print("Tradable Pairs Shape:", tradable_pairs.shape)

# -------------------------------------------------------------------
print("---------- PHASE 3: DEVIATION SIGNAL OPTIMIZATION ----------")
# -------------------------------------------------------------------

# ---------- CALCULATE MAX DEVIATIONS ----------

def max_deviation(tradable_residual_series, tradable_equilibrium_values):
    """
    Compute the maximum absolute deviation from equilibrium for each tradable pair.

    Args:
        tradable_residual_series: 2D array (n_pairs, time) of residuals.
        tradable_equilibrium_values: 1D array (n_pairs,) of equilibrium levels.

    Returns:
        1D array of max absolute deviations per pair.
    """
    deviations = tradable_residual_series - tradable_equilibrium_values[:, np.newaxis]
    abs_deviations = np.abs(deviations)
    max_abs_devs = np.max(abs_deviations, axis=1)
    return np.array(max_abs_devs)

max_deviations = max_deviation(tradable_residual_series, tradable_equilibrium_values)
print("Max Deviations Shape:", max_deviations.shape)

# ---------- CALCULATE DEVIATION RANGES ----------

def deviation_range(max_deviations):
    """
    Build a deviation grid from 0 to each pair's max deviation.

    Args:
        max_deviations: 1D array of max absolute deviations per pair.

    Returns:
        2D array (n_pairs, deviation_range_intervals) of deviation thresholds.
    """
    deviation_range = np.array([
        np.linspace(0, max_dev, deviation_range_intervals) for max_dev in max_deviations
    ])
    return deviation_range

deviation_ranges = deviation_range(max_deviations)
print("Deviation Ranges Shape:", deviation_ranges.shape)

# ---------- CALCULATE PLUS & MINUS DEVIATION RANGES ----------

deviation_plus_values = tradable_equilibrium_values[:, np.newaxis] + deviation_ranges
deviation_minus_values = tradable_equilibrium_values[:, np.newaxis] - deviation_ranges

# ---------- CALCULATE DEVIATION COUNTS & FREQUENCIES ----------

deviation_plus_count = np.array(np.sum(tradable_residual_series[:, :, np.newaxis] > deviation_plus_values[:, np.newaxis, :], axis=1))
deviation_minus_count = np.array(np.sum(tradable_residual_series[:, :, np.newaxis] < deviation_minus_values[:, np.newaxis, :], axis=1))

deviation_count_total = deviation_plus_count + deviation_minus_count
print("Deviation Count Total Shape:", deviation_count_total.shape)

deviation_frequency = (deviation_count_total / data_relevance_period) / 2
print("Deviation Frequency Shape:", deviation_frequency.shape)

# ---------- CALCULATE PROFIT FUNCTION ----------

profit_function = 2 * data_relevance_period * deviation_ranges * (deviation_frequency ** 2)

# ---------- ADJUST DEVIATION FREQUENCIES FOR STRICT MONOTONICITY ----------

def interpolate_for_monotonicity(deviation_frequency, deviation_ranges):
    """
    Enforce a strictly decreasing deviation frequency curve via interpolation.

    For each pair, keeps only points where the frequency decreases as deviation
    increases, then linearly interpolates back onto the full deviation grid.

    Args:
        deviation_frequency: 2D array (n_pairs, n_levels) of frequencies per level.
        deviation_ranges: 2D array (n_pairs, n_levels) of deviation thresholds.

    Returns:
        2D array (n_pairs, n_levels) of monotonic (decreasing) frequencies.
    """
    interpolated_series = []

    for i in range(deviation_frequency.shape[0]):
        freq_row = deviation_frequency[i]
        range_row = deviation_ranges[i]

        # Find strictly decreasing points
        decreasing_mask = np.diff(freq_row) < 0
        valid_indices = np.where(np.concatenate([[True], decreasing_mask]))[0]

        x_valid = range_row[valid_indices]
        y_valid = freq_row[valid_indices]

        # Remove any duplicate x values
        x_valid, unique_indices = np.unique(x_valid, return_index=True)
        y_valid = y_valid[unique_indices]

        interpolated = np.interp(range_row, x_valid, y_valid)

        interpolated_series.append(interpolated)

    return np.array(interpolated_series)

monotonic_deviation_frequency = interpolate_for_monotonicity(deviation_frequency, deviation_ranges)
print("Monotonic Deviation Frequency Shape:", monotonic_deviation_frequency.shape)

# ---------- CALCULATE MONOTONIC PROFIT FUNCTION ----------

monotonic_profit_function = 2 * data_relevance_period * deviation_ranges * (monotonic_deviation_frequency ** 2)

# ---------- PERFORM TIKHONOV-MILLER REGULARISATION ----------

def tikhonov_regularisation(freq, regularisation_lambda):
    """
    Smooth a 1D frequency curve using Tikhonov regularisation.

    Applies a second order difference penalty to reduce curvature while staying
    close to the original series, solving:
        (I + λ DᵀD) x = y

    Args:
        freq: 1D array of frequencies to smooth.
        regularisation_lambda: Smoothing strength (higher = smoother).

    Returns:
        1D array of smoothed frequencies.
    """
    n = len(freq)
    I = np.eye(n)

    # 2nd order difference operator
    D = np.diff(np.eye(n), n=2, axis=0)
    
    # Regularized least squares solution: (I + λ DᵀD)^-1 y
    smoothed = np.linalg.solve(I + regularisation_lambda * D.T @ D, freq)
    
    return smoothed

regularised_deviation_frequency = np.array([
    tikhonov_regularisation(freq_row, regularisation_lambda)
    for freq_row in deviation_frequency
])

print("Regularised Deviation Frequency Shape:", regularised_deviation_frequency.shape)

# ---------- CALCULATE REGULARISED PROFIT FUNCTION ----------

regularised_profit_function = 2 * data_relevance_period * deviation_ranges * (regularised_deviation_frequency ** 2)

# ---------- FIND OPTIMAL DEVIATION LEVELS ----------

optimal_deviation_index = np.argmax(regularised_profit_function, axis=1)
optimal_deviation = deviation_ranges[np.arange(deviation_ranges.shape[0]), optimal_deviation_index]

print("Optimal Deviation Shape:", optimal_deviation.shape)

optimal_deviation_plus = tradable_equilibrium_values + optimal_deviation
optimal_deviation_minus = tradable_equilibrium_values - optimal_deviation

# ------------------------------------------------------------
print("---------- PHASE 4: PROFITABILITY BACKTEST ----------")
# ------------------------------------------------------------

# ---------- FIND TRADABLE PAIR LOG PRICES ----------

B_indices = tradable_pairs[:, 0]
A_indices = tradable_pairs[:, 1]

tradable_prices_B = price_data[:, B_indices]
tradable_prices_A = price_data[:, A_indices]

relevant_tradable_prices_B = price_data[:90, B_indices]
relevant_tradable_prices_A = price_data[:90, A_indices]

tradable_log_prices_B = log_price_data[:, B_indices]
tradable_log_prices_A = log_price_data[:, A_indices]

print("Relevant Tradable Prices (B):",relevant_tradable_prices_B.shape)
print("Relevant Tradable Prices (A):",relevant_tradable_prices_A.shape)

# ---------- SETUP BACKTEST DICTIONARIES ----------

backtest_portfolio_1 = {}
backtest_portfolio_2 = {}

backtest_sin_bin_1 = []
backtest_sin_bin_2 = []

backtest_cumulative_return_records = {}
backtest_incremental_return_records = {i: [] for i in range(tradable_pairs.shape[0])}

backtest_long_entry_points = {i: [] for i in range(tradable_pairs.shape[0])}
backtest_long_exit_points = {i: [] for i in range(tradable_pairs.shape[0])}
backtest_long_stop_loss_points = {i: [] for i in range(tradable_pairs.shape[0])}

backtest_short_entry_points = {i: [] for i in range(tradable_pairs.shape[0])}
backtest_short_exit_points = {i: [] for i in range(tradable_pairs.shape[0])}
backtest_short_stop_loss_points = {i: [] for i in range(tradable_pairs.shape[0])}

# ---------- BACKTEST LOOP ----------

for day in range(data_relevance_period):

    # --- COMPUTE CURRENT DATA ---

    current_log_prices_B = tradable_log_prices_B[day]
    current_log_prices_A = tradable_log_prices_A[day]
    current_residuals = tradable_residual_series[:, day]

    # ----- EXITS -----

    # --- TRADE 1 EXIT (LONG SPREAD) ---

    for tradable_pair in list(backtest_portfolio_1.keys()):

        current_log_price_B = current_log_prices_B[tradable_pair]
        current_log_price_A = current_log_prices_A[tradable_pair]
        current_residual = current_residuals[tradable_pair]
        stored_log_price_A, stored_log_price_B, open_day = backtest_portfolio_1[tradable_pair]
        stop_loss_day = open_day + (reversion_time_multiplier * tradable_reversion_times[tradable_pair])

        # --- Exit ---
            
        if current_residual >= tradable_equilibrium_values[tradable_pair]:

            if tradable_pair in backtest_sin_bin_1:
                continue

            return_A1 = current_log_price_A - stored_log_price_A
            return_B1 = stored_log_price_B - current_log_price_B
            total_return_1 = return_A1 + (np.abs(tradable_cointegration_coefficients[tradable_pair]) * return_B1)

            if tradable_pair not in backtest_cumulative_return_records:
                backtest_cumulative_return_records[tradable_pair] = 0.0
            backtest_cumulative_return_records[tradable_pair] += total_return_1

            backtest_long_exit_points[tradable_pair].append(day)
            backtest_incremental_return_records[tradable_pair].append(total_return_1)

            del backtest_portfolio_1[tradable_pair]

        # --- Stop Loss ---

        else:
            
            if tradable_pair in backtest_sin_bin_1:
                continue

            if day > stop_loss_day:

                return_A1_SL = current_log_price_A - stored_log_price_A
                return_B1_SL = stored_log_price_B - current_log_price_B
                total_return_1_SL = return_A1_SL + (np.abs(tradable_cointegration_coefficients[tradable_pair]) * return_B1_SL)

                if tradable_pair not in backtest_cumulative_return_records:
                    backtest_cumulative_return_records[tradable_pair] = 0.0
                backtest_cumulative_return_records[tradable_pair] += total_return_1_SL

                backtest_long_stop_loss_points[tradable_pair].append(day)
                backtest_incremental_return_records[tradable_pair].append(total_return_1_SL)

                backtest_sin_bin_1.append(tradable_pair)
                del backtest_portfolio_1[tradable_pair]

    # --- TRADE 2 EXIT (SHORT SPREAD) ---

    for tradable_pair in list(backtest_portfolio_2.keys()):

        current_log_price_B = current_log_prices_B[tradable_pair]
        current_log_price_A = current_log_prices_A[tradable_pair]
        current_residual = current_residuals[tradable_pair]
        stored_log_price_A, stored_log_price_B, open_day = backtest_portfolio_2[tradable_pair]
        stop_loss_day = open_day + (reversion_time_multiplier * tradable_reversion_times[tradable_pair])

        # --- Exit ---

        if current_residual <= tradable_equilibrium_values[tradable_pair]:

            if tradable_pair in backtest_sin_bin_2:
                continue

            return_A2 = stored_log_price_A - current_log_price_A
            return_B2 = current_log_price_B - stored_log_price_B
            total_return_2 = return_A2 + (np.abs(tradable_cointegration_coefficients[tradable_pair]) * return_B2)

            if tradable_pair not in backtest_cumulative_return_records:
                backtest_cumulative_return_records[tradable_pair] = 0.0
            backtest_cumulative_return_records[tradable_pair] += total_return_2

            backtest_short_exit_points[tradable_pair].append(day)
            backtest_incremental_return_records[tradable_pair].append(total_return_2)

            del backtest_portfolio_2[tradable_pair]

        # --- Stop Loss ---

        else:
            
            if tradable_pair in backtest_sin_bin_2:
                continue

            if day > stop_loss_day:

                return_A2_SL = stored_log_price_A - current_log_price_A
                return_B2_SL = current_log_price_B - stored_log_price_B
                total_return_2_SL = return_A2_SL + (np.abs(tradable_cointegration_coefficients[tradable_pair]) * return_B2_SL)

                if tradable_pair not in backtest_cumulative_return_records:
                    backtest_cumulative_return_records[tradable_pair] = 0.0
                backtest_cumulative_return_records[tradable_pair] += total_return_2_SL

                backtest_short_stop_loss_points[tradable_pair].append(day)
                backtest_incremental_return_records[tradable_pair].append(total_return_2_SL)

                backtest_sin_bin_2.append(tradable_pair)
                del backtest_portfolio_2[tradable_pair]

    # ----- SIN BIN 1 -----

    for tradable_pair in range(tradable_pairs.shape[0]):

        current_residual = current_residuals[tradable_pair]

        if tradable_pair in backtest_sin_bin_1:

            if current_residual >= tradable_equilibrium_values[tradable_pair]:

                backtest_sin_bin_1.remove(tradable_pair)

    # ----- SIN BIN 2 -----

    for tradable_pair in range(tradable_pairs.shape[0]):

        current_residual = current_residuals[tradable_pair]

        if tradable_pair in backtest_sin_bin_2:

            if current_residual <= tradable_equilibrium_values[tradable_pair]:

                backtest_sin_bin_2.remove(tradable_pair)

    # ----- ENTRIES -----

    for tradable_pair in range(tradable_pairs.shape[0]):

        current_log_price_B = current_log_prices_B[tradable_pair]
        current_log_price_A = current_log_prices_A[tradable_pair]
        current_residual = current_residuals[tradable_pair]

        # --- TRADE 1 ENTRY (LONG SPREAD) ---
        if current_residual <= optimal_deviation_minus[tradable_pair]:

            if tradable_pair in backtest_portfolio_1:
              continue

            if tradable_pair in backtest_sin_bin_1:
              continue

            backtest_portfolio_1[tradable_pair] = (current_log_price_A, current_log_price_B, day)

            backtest_long_entry_points[tradable_pair].append(day)

        # --- TRADE 2 ENTRY (SHORT SPREAD) ---

        if current_residual >= optimal_deviation_plus[tradable_pair]:

            if tradable_pair in backtest_portfolio_2:
              continue

            if tradable_pair in backtest_sin_bin_2:
              continue

            backtest_portfolio_2[tradable_pair] = (current_log_price_A, current_log_price_B, day)

            backtest_short_entry_points[tradable_pair].append(day)
 
# ---------- CALCULATE SHARPE RATIOS ----------

sharpe_ratios = []
backtested_cumulative_returns = []

for k in backtest_incremental_return_records:

    returns = backtest_incremental_return_records[k]
    cumulative_returns = np.sum(returns)

    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    if std_dev > 0:
        ratio = (mean_return - rf) / std_dev
    
    sharpe_ratios.append(ratio)
    backtested_cumulative_returns.append(cumulative_returns)

sharpe_ratios = np.array(sharpe_ratios)
backtested_cumulative_returns = np.array(backtested_cumulative_returns)

print("Sharpe Ratios Shape:",sharpe_ratios.shape)

# ---------- SORT BY SHARPE RATIOS ----------

backtested_indices = np.argsort(sharpe_ratios)[::-1][:number_of_pairs_selected_from_backtest]
backtested_pairs = tradable_pairs[backtested_indices]

print(f"Top 5 Tradable Pairs by Sharpe Ratio ({data_relevance_period} Period):\n")

for idx in backtested_indices:
    asset_A, asset_B = tradable_pairs[idx]
    original_pair_index = tradable_pair_index[idx]
    print(f"Pair {original_pair_index} (Assets {asset_A} & {asset_B}): {sharpe_ratios[idx]:.4f} ({backtested_cumulative_returns[idx] * 100:.2f}% Return)")

# ---------- PLOT FUNCTIONS ----------

def basic_plot(data,title,xlabel,ylabel):
    """
    Plot a simple line chart with title, axis labels, and a grid.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def tradable_pairs_plot(tradable_pair_index, residual_series, equilibrium_values):
    """
    Plot residual series with equilibrium lines for selected tradable pairs.

    Args:
        tradable_pair_index: Iterable of pair indices to plot.
        residual_series: 2D array (n_pairs, time) of residuals.
        equilibrium_values: 1D array (n_pairs,) of equilibrium levels.
    """
    for i in tradable_pair_index:
        plt.figure(figsize=(15, 7))
        plt.plot(residual_series[i], label=f"Residual: Pair {i}", color='black')
        plt.axhline(equilibrium_values[i], color='gray', linestyle='--')
        plt.title(f"Residual Series for Pair {i}")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.legend()
        plt.grid(True)
        plt.show()

def deviation_frequency_plot(deviation_ranges, deviation_frequency, regularised_frequency):
    """
    Plot original vs regularised deviation frequency curves for each tradable pair.

    Note:
        Uses the global `tradable_pair_index` to map each row back to the original
        pair index for labeling.
    """
    for i in range(deviation_frequency.shape[0]):
        original_index = tradable_pair_index[i]

        plt.figure(figsize=(15, 7))
        plt.plot(deviation_ranges[i], deviation_frequency[i], label='Original', linestyle='-', color = 'green')
        plt.plot(deviation_ranges[i], regularised_frequency[i], label='Regularised', linestyle='-', color = 'blue')
        plt.title(f'Deviation Frequency for Pair {original_index}')
        plt.xlabel('Deviation From Equilibrium')
        plt.ylabel('Deviation Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def profit_function_plot(deviation_ranges, profit_function, regularised_profit_function):
    """
    Plot original vs regularised profit curves across deviation thresholds.

    Note:
        Uses the global `tradable_pair_index` to map each row back to the original
        pair index for labeling.
    """
    for i in range(profit_function.shape[0]):
        original_index = tradable_pair_index[i]

        plt.figure(figsize=(15, 7))
        plt.plot(deviation_ranges[i], profit_function[i], label='Original', linestyle='-', color = 'green')
        plt.plot(deviation_ranges[i], regularised_profit_function[i], label='Regularised', linestyle='-', color = 'blue')
        plt.title(f'Profit Value for Pair {original_index}')
        plt.xlabel('Deviation From Equilibrium')
        plt.ylabel('Profit Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def deviation_signals_plot(tradable_pair_index, residual_series, equilibrium_values, optimal_deviation_plus, optimal_deviation_minus):
    """
    Plot residuals with equilibrium and optimal entry/exit deviation bands.

    Args:
        tradable_pair_index: Iterable of original pair indices to plot.
        residual_series: 2D array (n_pairs, time) of residuals.
        equilibrium_values: 1D array (n_pairs,) of equilibrium levels.
        optimal_deviation_plus: 1D array of upper deviation thresholds (aligned to tradable_pair_index order).
        optimal_deviation_minus: 1D array of lower deviation thresholds (aligned to tradable_pair_index order).
    """
    for idx, i in enumerate(tradable_pair_index):
        plt.figure(figsize=(15, 7))
        plt.plot(residual_series[i], label=f"Residual: Pair {i}", color='black')
        plt.axhline(equilibrium_values[i], color='gray', linestyle='--')

        plt.axhline(optimal_deviation_plus[idx], color='green', linestyle='--')
        plt.axhline(optimal_deviation_minus[idx], color='green', linestyle='--')

        plt.title(f"Residual Series for Pair {i}")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.legend()
        plt.grid(True)
        plt.show()

def profitability_backtest_plot(tradable_pair_index, residual_series, equilibrium_values, optimal_deviation_plus, optimal_deviation_minus, backtest_cumulative_return_records, tradable_pairs, tradable_cointegration_coefficients):
    """
    Plot residuals for each tradable pair with backtest trade markers and performance stats.

    Overlays the residual series and equilibrium level, plus deviation thresholds and the
    backtest event points (entries, exits, and stop losses). The title is annotated with
    the asset pair, sharpe ratio (from global `sharpe_ratios`), total return over the
    `data_relevance_period`, and the cointegration coefficient.

    Note:
        Uses global arrays for backtest point locations:
        `backtest_long_entry_points`, `backtest_short_entry_points`,
        `backtest_long_exit_points`, `backtest_short_exit_points`,
        `backtest_long_stop_loss_points`, `backtest_short_stop_loss_points`,
        plus `sharpe_ratios` and `data_relevance_period`.

    Args:
        tradable_pair_index: Iterable of original pair indices for labeling.
        residual_series: 2D array (n_pairs, time) of residuals.
        equilibrium_values: 1D array (n_pairs,) equilibrium levels.
        optimal_deviation_plus: 1D array of upper thresholds (aligned to tradable order).
        optimal_deviation_minus: 1D array of lower thresholds (aligned to tradable order).
        backtest_cumulative_return_records: 1D array of cumulative returns per pair.
        tradable_pairs: Array-like of (asset_A, asset_B) identifiers per tradable pair.
        tradable_cointegration_coefficients: 1D array of coefficients per tradable pair.
    """
    for idx, i in enumerate(tradable_pair_index):
        plt.figure(figsize=(15, 7))
        plt.plot(residual_series[i], label=f"Residual: Pair {i}", color='black')
        plt.axhline(equilibrium_values[i], color='gray', linestyle='--', label='Equilibrium')

        # --- Entry & Exit Signals ---

        long_entry_points = backtest_long_entry_points[idx]
        plt.scatter(long_entry_points, residual_series[i][long_entry_points], color='green', marker='o', label='Entry Long', zorder=5)

        short_entry_points = backtest_short_entry_points[idx]
        plt.scatter(short_entry_points, residual_series[i][short_entry_points], color='blue', marker='o', label='Entry Short', zorder=5)

        long_exit_points = backtest_long_exit_points[idx]
        plt.scatter(long_exit_points, residual_series[i][long_exit_points], color='green', marker='x', label='Exit Long', zorder=5)

        short_exit_points = backtest_short_exit_points[idx]
        plt.scatter(short_exit_points, residual_series[i][short_exit_points], color='blue', marker='x', label='Exit Short', zorder=5)

        long_stop_loss_points = backtest_long_stop_loss_points[idx]
        plt.scatter(long_stop_loss_points, residual_series[i][long_stop_loss_points], color='red', marker='*', label='Stop Loss Long', zorder=5)

        short_stop_loss_points = backtest_short_stop_loss_points[idx]
        plt.scatter(short_stop_loss_points, residual_series[i][short_stop_loss_points], color='red', marker='*', label='Stop Loss Short', zorder=5)

        # ----------------------------

        plt.axhline(optimal_deviation_plus[idx], color='grey', linestyle='--', label='Deviation Thresholds')
        plt.axhline(optimal_deviation_minus[idx], color='grey', linestyle='--')

        # --- Get asset info and return % ---

        asset_A, asset_B = tradable_pairs[idx]
        return_percent = float(backtest_cumulative_return_records[idx]) * 100

        # --- Update title ---
        plt.title(f"Residual Series for Pair {i} (Assets {asset_A} & {asset_B}) | Sharpe Ratio: {sharpe_ratios[idx]:.2f} | {data_relevance_period} Period Return: {return_percent:.2f}% | λ: {tradable_cointegration_coefficients[idx]:.2f}", fontsize=14)

        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.legend()
        plt.grid(True)
        plt.show()

def raw_price_plot_backtest(pairs):
    """
    Plot raw price series for each tradable pair with backtest trade markers.

    Creates a 1x2 plot per pair showing Asset A and Asset B prices side by side,
    overlaying entry, exit, and stop loss points from the backtest.

    Note:
        Relies on global arrays:
        `relevant_tradable_prices_A`, `relevant_tradable_prices_B`,
        `backtest_long_entry_points`, `backtest_short_entry_points`,
        `backtest_long_exit_points`, `backtest_short_exit_points`,
        `backtest_long_stop_loss_points`, `backtest_short_stop_loss_points`,
        and `tradable_pair_index`.

    Args:
        pairs: Array like of (stock_B, stock_A) identifiers for each tradable series.
    """
    num_series = relevant_tradable_prices_A.shape[1]

    for i in range(num_series):
        stock_B, stock_A = pairs[i]

        fig, axs = plt.subplots(1, 2, figsize=(15, 4))

        # --- Series A ---
        axs[0].plot(relevant_tradable_prices_A[:, i])
        axs[0].scatter(backtest_long_entry_points[i], relevant_tradable_prices_A[:, i][backtest_long_entry_points[i]], color='green', marker='o', label='Entry Long', zorder=5)
        axs[0].scatter(backtest_short_entry_points[i], relevant_tradable_prices_A[:, i][backtest_short_entry_points[i]], color='blue', marker='o', label='Entry Short', zorder=5)

        axs[0].scatter(backtest_long_exit_points[i], relevant_tradable_prices_A[:, i][backtest_long_exit_points[i]], color='green', marker='x', label='Exit Long', zorder=5)
        axs[0].scatter(backtest_short_exit_points[i], relevant_tradable_prices_A[:, i][backtest_short_exit_points[i]], color='blue', marker='x', label='Exit Short', zorder=5)

        axs[0].scatter(backtest_long_stop_loss_points[i], relevant_tradable_prices_A[:, i][backtest_long_stop_loss_points[i]], color='red', marker='*', label='Stop Loss Long', zorder=5)
        axs[0].scatter(backtest_short_stop_loss_points[i], relevant_tradable_prices_A[:, i][backtest_short_stop_loss_points[i]], color='red', marker='*', label='Stop Loss Short', zorder=5)

        axs[0].set_title(f"Raw Price for Asset {stock_A} (A in Pair {tradable_pair_index[i]})")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Price")
        axs[0].grid(True)
        axs[0].legend()

        # --- Series B ---
        axs[1].plot(relevant_tradable_prices_B[:, i])
        axs[1].scatter(backtest_long_entry_points[i], relevant_tradable_prices_B[:, i][backtest_long_entry_points[i]], color='blue', marker='o', label='Entry Short', zorder=5)
        axs[1].scatter(backtest_short_entry_points[i], relevant_tradable_prices_B[:, i][backtest_short_entry_points[i]], color='green', marker='o', label='Entry Long', zorder=5)

        axs[1].scatter(backtest_long_exit_points[i], relevant_tradable_prices_B[:, i][backtest_long_exit_points[i]], color='blue', marker='x', label='Exit Short', zorder=5)
        axs[1].scatter(backtest_short_exit_points[i], relevant_tradable_prices_B[:, i][backtest_short_exit_points[i]], color='green', marker='x', label='Exit Long', zorder=5)

        axs[1].scatter(backtest_long_stop_loss_points[i], relevant_tradable_prices_B[:, i][backtest_long_stop_loss_points[i]], color='red', marker='*', label='Stop Loss Short', zorder=5)
        axs[1].scatter(backtest_short_stop_loss_points[i], relevant_tradable_prices_B[:, i][backtest_short_stop_loss_points[i]], color='red', marker='*', label='Stop Loss Long', zorder=5)

        axs[1].set_title(f"Raw Price for Asset {stock_B} (B in Pair {tradable_pair_index[i]})")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Price")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

# ---------- PLOTS ----------

#basic_plot(price_data, title = "Price Data", xlabel = "Time", ylabel = "Price")

#basic_plot(price_data[:90,28], title = "Price Data (A)", xlabel = "Time", ylabel = "Price")
#basic_plot(price_data[:90,6], title = "Price Data (B)", xlabel = "Time", ylabel = "Price")

#tradable_pairs_plot(tradable_pair_index,residual_series,equilibrium_values)

#deviation_frequency_plot(deviation_ranges, deviation_frequency, regularised_deviation_frequency)

#profit_function_plot(deviation_ranges, profit_function, regularised_profit_function)

#deviation_signals_plot(tradable_pair_index, residual_series, equilibrium_values, optimal_deviation_plus, optimal_deviation_minus)

#profitability_backtest_plot(tradable_pair_index, residual_series, equilibrium_values, optimal_deviation_plus, optimal_deviation_minus, backtest_cumulative_return_records, tradable_pairs,tradable_cointegration_coefficients)

#raw_price_plot_backtest(tradable_pairs)

# ---------- DISCLAIMERS ----------

# Slippage & trading fees are assumed to be 0 in this version
# Price is assumed as VWAP in this version (price at max volume within time period) - important (P 112)
# Bootstrapping reversion time distrubution and evaulating tradability on percentiles has not been used in this version
# Only market neutral pairs are deemed tradable in this version of PAIRZILLA (+ve cointegration coefficients, long & short positions only)
# Stop Loss signal does not cover deviation to deviation trading, only deviation to EQ
