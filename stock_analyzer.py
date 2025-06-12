#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Stock Analyzer - Core Analysis Engine
"""

import logging

# Configure logger early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.critical("Pandas library not found. Please install it by running: pip install pandas")
    pd = None # Placeholder to allow script to run, functionality will be affected

try:
    import numpy as np
except ImportError:
    logger.critical("NumPy library not found. Please install it by running: pip install numpy")
    np = None # Placeholder

try:
    import yfinance as yf
except ImportError:
    logger.critical("yfinance library not found. Please install it by running: pip install yfinance")
    yf = None # Placeholder

try:
    import requests
except ImportError:
    logger.critical("Requests library not found. Please install it by running: pip install requests")
    requests = None # Placeholder

try:
    import ta
except ImportError:
    logger.critical("Technical Analysis (ta) library not found. Please install it by running: pip install ta")
    ta = None # Placeholder

import time # Added for retry delay
from typing import Dict, List, Optional, Tuple, Union # Added Union
from datetime import datetime
from dataclasses import dataclass
from config import Config

@dataclass
class StockAnalysis:
    ticker: str
    company_name: str
    current_price: float
    sector: str
    rsi: float
    sma_20: float
    sma_50: float
    momentum_20d: float
    pe_ratio: float
    roe: float
    debt_to_equity: float
    technical_score: float
    fundamental_score: float
    overall_score: float
    recommendation: str
    confidence: float
    reasons: List[str]
    # New fields for breakout signals
    is_volume_breakout: Optional[bool] = None
    volume_ratio: Optional[float] = None
    # MA Crossover signals
    sma_20_50_crossed_bullish: Optional[bool] = None
    sma_50_200_crossed_bullish: Optional[bool] = None
    sma_20_50_crossed_bearish: Optional[bool] = None
    sma_50_200_crossed_bearish: Optional[bool] = None
    # Bollinger Band signals
    bb_breakout_upper: Optional[bool] = None
    bb_squeeze: Optional[bool] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_ma: Optional[float] = None
    # N-day High/Low breakout signals
    breakout_n_day_high: Optional[bool] = None
    breakdown_n_day_low: Optional[bool] = None
    n_day_high: Optional[float] = None
    n_day_low: Optional[float] = None
    # MACD Crossover and RSI Recovery
    macd_bullish_crossover: Optional[bool] = None
    macd_bearish_crossover: Optional[bool] = None
    rsi_recovered_from_oversold: Optional[bool] = None
    macd_line: Optional[float] = None
    macd_signal_line: Optional[float] = None
    # Breakout score
    breakout_score: Optional[float] = None
    breakout_reasons: Optional[List[str]] = None
    # Trading plan fields
    suggested_entry_price: Optional[float] = None
    target_price_1: Optional[float] = None
    stop_loss_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    risk_level: Optional[str] = None  # e.g., "Conservative", "Moderate", "Aggressive"
    max_position_size_pct: Optional[float] = None # e.g., 0.02 for 2%
    trading_strategy_summary: Optional[str] = None
    time_horizon: Optional[str] = None


class StockAnalyzer:
    def __init__(self):
        self.config = Config()
        if pd and np and yf and requests and ta: # Check if all were imported
            logger.info("All major libraries loaded successfully.")
        else:
            logger.warning("One or more critical libraries failed to load. Functionality may be limited.")
        logger.info("Stock Analyzer initialized")

    def calculate_trading_parameters(self, data: Dict, hist_data: Optional[pd.DataFrame]=None) -> Dict:
        """
        Calculates suggested trading parameters based on risk level and breakout score.
        'hist_data' is included for ATR calculations.
        """
        trading_params: Dict[str, Optional[Union[float, str]]] = {
            'suggested_entry_price': None,
            'target_price_1': None,
            'stop_loss_price': None,
            'risk_reward_ratio': None,
            'risk_level': "Undefined",
            'max_position_size_pct': None,
            'trading_strategy_summary': "Strategy not determined.", # Default summary
            'current_atr': None,
            'time_horizon': None # Initialize time_horizon
        }

        ticker = data.get('ticker', 'Unknown')
        breakout_score_val = data.get('breakout_score')
        current_price = data.get('current_price')

        if breakout_score_val is None or current_price is None:
            logger.warning(f"Ticker '{ticker}': Breakout score ({breakout_score_val}) or current price ({current_price}) is missing. Cannot calculate trading parameters.")
            trading_params['trading_strategy_summary'] = "Breakout score or current price missing for parameter calculation."
            return trading_params

        cfg_trading = self.config.TradingSettings
        risk_level_str = "Undefined"
        stop_loss_pct = None
        target_profit_pct = None
        max_pos_pct = None

        if breakout_score_val >= cfg_trading.CONSERVATIVE_BREAKOUT_SCORE_MIN:
            risk_level_str = "Conservative"
            stop_loss_pct = cfg_trading.CONSERVATIVE_STOP_LOSS_PCT
            target_profit_pct = cfg_trading.CONSERVATIVE_TARGET_PROFIT_PCT
            max_pos_pct = cfg_trading.CONSERVATIVE_MAX_POS_SIZE_PCT
        elif breakout_score_val >= cfg_trading.MODERATE_BREAKOUT_SCORE_MIN:
            risk_level_str = "Moderate"
            stop_loss_pct = cfg_trading.MODERATE_STOP_LOSS_PCT
            target_profit_pct = cfg_trading.MODERATE_TARGET_PROFIT_PCT
            max_pos_pct = cfg_trading.MODERATE_MAX_POS_SIZE_PCT
        elif breakout_score_val >= cfg_trading.AGGRESSIVE_BREAKOUT_SCORE_MIN:
            risk_level_str = "Aggressive"
            stop_loss_pct = cfg_trading.AGGRESSIVE_STOP_LOSS_PCT
            target_profit_pct = cfg_trading.AGGRESSIVE_TARGET_PROFIT_PCT
            max_pos_pct = cfg_trading.AGGRESSIVE_MAX_POS_SIZE_PCT
        else:
            logger.info(f"Ticker '{ticker}': Breakout score {breakout_score_val:.1f} is below defined thresholds for a trade.")
            trading_params['risk_level'] = risk_level_str # Will be "Undefined"
            trading_params['trading_strategy_summary'] = f"No actionable trade setup: Breakout score {breakout_score_val:.1f} too low."
            return trading_params

        trading_params['risk_level'] = risk_level_str
        trading_params['max_position_size_pct'] = max_pos_pct
        trading_params['time_horizon'] = cfg_trading.DEFAULT_TRADE_TIME_HORIZON # Assign time horizon for valid trades

        suggested_entry = current_price
        trading_params['suggested_entry_price'] = suggested_entry

        if stop_loss_pct is not None and suggested_entry is not None:
            trading_params['stop_loss_price'] = suggested_entry * (1 - stop_loss_pct)

        if target_profit_pct is not None and suggested_entry is not None:
            trading_params['target_price_1'] = suggested_entry * (1 + target_profit_pct)

        sl_price = trading_params['stop_loss_price']
        tp_price = trading_params['target_price_1']

        rr_ratio_val = None # Local variable for R/R
        if suggested_entry is not None and isinstance(sl_price, (float, int)) and isinstance(tp_price, (float, int)):
            potential_reward = tp_price - suggested_entry
            potential_risk = suggested_entry - sl_price
            if potential_risk > 0.0001:
                rr_ratio_val = potential_reward / potential_risk
                trading_params['risk_reward_ratio'] = rr_ratio_val
            else:
                logger.warning(f"Ticker '{ticker}': Potential risk is zero or negative. R/R not calculated. Entry: {suggested_entry}, SL: {sl_price}")

        # ATR Calculation
        current_atr_val = None # Local variable for ATR
        if ta and hist_data is not None and not hist_data.empty and \
           all(col in hist_data.columns for col in ['High', 'Low', 'Close']):
            try:
                if len(hist_data['Close']) >= cfg_trading.ATR_PERIOD:
                    atr_indicator = ta.volatility.AverageTrueRange(
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close'],
                        window=cfg_trading.ATR_PERIOD
                    )
                    calculated_atr = atr_indicator.average_true_range().iloc[-1]
                    if pd.notna(calculated_atr):
                         trading_params['current_atr'] = calculated_atr
                         current_atr_val = calculated_atr # For summary string
                         logger.debug(f"Ticker '{ticker}': Calculated ATR({cfg_trading.ATR_PERIOD}) = {current_atr_val:.2f}")
                else:
                    logger.debug(f"Ticker '{ticker}': Not enough data for ATR({cfg_trading.ATR_PERIOD}) calculation.")
            except Exception as e_atr:
                logger.error(f"Ticker '{ticker}': Error calculating ATR: {e_atr}", exc_info=True)

        # Refined Trading Strategy Summary
        summary = f"Strategy: {risk_level_str}"
        if suggested_entry is not None: summary += f" | Entry: Market ~${suggested_entry:.2f} on breakout confirmation"
        if sl_price is not None: summary += f" | SL: ~${sl_price:.2f}"
        if tp_price is not None: summary += f" | TP1: ~${tp_price:.2f}"
        if rr_ratio_val is not None: summary += f" | R/R: {rr_ratio_val:.2f}:1"
        if max_pos_pct is not None: summary += f" | Max Pos Size: {max_pos_pct:.0%}"
        if trading_params['time_horizon'] is not None: summary += f" | Horizon: {trading_params['time_horizon']}"
        # Could add ATR to summary if desired:
        # if current_atr_val is not None: summary += f" | ATR({cfg_trading.ATR_PERIOD}): {current_atr_val:.2f}"

        trading_params['trading_strategy_summary'] = summary
        logger.info(f"Ticker '{ticker}': Trading parameters. Summary: {summary}")

        return trading_params

    def get_sp500_tickers(self) -> List[str]:
        try:
            if not pd:
                logger.error("Pandas is not available, cannot retrieve S&P 500 tickers. Using fallback list.")
                return self.config.FALLBACK_TICKERS # Assuming FALLBACK_TICKERS is defined in Config
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url) # type: ignore
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist() # type: ignore
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving S&P 500 list: {e}. Using fallback list.")
            return self.config.FALLBACK_TICKERS # Assuming FALLBACK_TICKERS is defined in Config


    def get_stock_data(self, ticker: str) -> Optional[Dict]:
        # Initial basic ticker validation
        if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
            logger.warning(f"Invalid ticker format: Ticker is None, not a string, or empty. Value: '{ticker}'")
            return None

        original_ticker = ticker # Keep original for some logging if needed
        ticker = ticker.strip().upper() # Normalize

        if len(ticker) > 10: # Arbitrary reasonable max length for typical stock tickers
            logger.warning(f"Invalid ticker format: Ticker '{ticker}' (Original: '{original_ticker}') seems too long.")
            return None

        # Check for required libraries after basic ticker validation
        if not yf:
            logger.error(f"yfinance library is not available. Cannot get data for ticker '{ticker}'.")
            return None
        if not pd:
            logger.error(f"Pandas library is not available. Cannot process data for ticker '{ticker}'.")
            return None
        if not np:
            logger.error(f"NumPy library is not available. Cannot process data for ticker '{ticker}'.")
            return None

        logger.debug(f"Attempting to fetch data for ticker: {ticker} (Original: {original_ticker})")
        hist = None
        info = None
        stock = None
        last_exception = None

        for attempt in range(self.config.RETRY_ATTEMPTS):
            try:
                stock = yf.Ticker(ticker)
                # Accessing .info can itself be a network call or raise issues
                info = stock.info

                # Check if info is populated sufficiently
                # Using 'currency' or 'financialCurrency' as another check for a valid company object from yfinance
                if not info or info.get('regularMarketPrice') is None or info.get('financialCurrency') is None :
                    logger.warning(f"Ticker '{ticker}': No valid detailed info (e.g., market price, currency) found on attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}. Ticker might be delisted, invalid, or not a standard stock type.")
                    if attempt == self.config.RETRY_ATTEMPTS - 1: # Last attempt
                        logger.error(f"Ticker '{ticker}': Failed to get valid detailed info after {self.config.RETRY_ATTEMPTS} attempts. Aborting for this ticker.")
                        return None # Give up on this ticker if info is consistently bad

                hist = stock.history(period="3mo", timeout=self.config.API_TIMEOUT) # type: ignore

                if hist is not None and not hist.empty:
                    if len(hist) >= self.config.MIN_DATA_POINTS:
                        logger.info(f"Ticker '{ticker}': Successfully fetched history data on attempt {attempt + 1}.")
                        break  # Success
                    else: # History fetched but not enough data points
                        logger.warning(f"Ticker '{ticker}': Insufficient data points ({len(hist)}) on attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}. Need {self.config.MIN_DATA_POINTS}.")
                        # This will fall through to the final check after the loop.
                else: # History is None or empty
                    logger.warning(f"Ticker '{ticker}': No history data returned on attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}.")

                if attempt < self.config.RETRY_ATTEMPTS - 1: # Don't sleep on the last attempt
                    logger.info(f"Ticker '{ticker}': Retrying...")
                    time.sleep(self.config.RETRY_DELAY_SECONDS) # Use a configurable delay

            except Exception as e: # Catching a broader exception during yf.Ticker or stock.history
                last_exception = e
                logger.warning(f"Ticker '{ticker}': Error during data fetching (attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}): {e}. Retrying...")
                if attempt < self.config.RETRY_ATTEMPTS - 1:
                    time.sleep(self.config.RETRY_DELAY_SECONDS)

        # Post-loop checks
        if hist is None or hist.empty or len(hist) < self.config.MIN_DATA_POINTS:
            logger.error(f"Ticker '{ticker}': Failed to fetch sufficient history after {self.config.RETRY_ATTEMPTS} attempts. Min points: {self.config.MIN_DATA_POINTS}, Got: {len(hist) if hist is not None else 0}. Last error: {last_exception}")
            return None

        # Ensure info was successfully fetched and is still considered valid
        if not info or info.get('regularMarketPrice') is None or info.get('financialCurrency') is None:
             logger.error(f"Ticker '{ticker}': Valid detailed info could not be confirmed even if history was available. Cannot proceed.")
             return None

        try:
            # Safe data access
            if 'Close' not in hist.columns or hist['Close'].empty:
                logger.warning(f"Ticker '{ticker}': No 'Close' price data in history.")
                return None
            
            current_price = hist['Close'].iloc[-1] # type: ignore

            # Ensure enough data for rolling calculations
            min_len_for_sma20 = 20
            min_len_for_sma50 = 50
            min_len_for_rsi = 15 # 14 periods + 1 for diff
            min_len_for_momentum = 21

            sma_20 = hist['Close'].rolling(window=min_len_for_sma20).mean().iloc[-1] if len(hist['Close']) >= min_len_for_sma20 else None
            sma_50 = hist['Close'].rolling(window=min_len_for_sma50).mean().iloc[-1] if len(hist['Close']) >= min_len_for_sma50 else (sma_20 if sma_20 is not None else None) # Fallback to sma_20 or None

            rsi = None
            if len(hist['Close']) >= min_len_for_rsi:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean().iloc[-1]

                if np and (np.isnan(gain) or np.isnan(loss)): # Should not happen if hist['Close'] is clean
                    rsi = np.nan # Or None
                elif loss == 0:
                    if gain == 0: # No change in price over the period
                        rsi = 50.0  # Neutral RSI
                    else: # All gain, no loss
                        rsi = 100.0
                elif gain == 0: # All loss, no gain
                    rsi = 0.0
                else:
                    rs = gain / loss
                    if np and (np.isnan(rs) or np.isinf(rs)):
                        rsi = np.nan # Or None, if rs is problematic
                    else:
                        rsi = 100.0 - (100.0 / (1.0 + rs))
            else: # Not enough data for RSI
                rsi = None # Or np.nan if preferred for consistency in numeric arrays

            momentum_20d = ((current_price / hist['Close'].iloc[-min_len_for_momentum]) - 1) * 100 if len(hist['Close']) >= min_len_for_momentum and current_price is not None else None

            # Convert np.nan to None for JSON compatibility and consistent handling, if np was used for rsi
            if np and isinstance(rsi, float) and np.isnan(rsi):
                rsi = None

            # Check if any essential calculated values are None (current_price and rsi are most critical for this check)
            if current_price is None or rsi is None: # type: ignore
                 logger.error(f"Ticker '{ticker}': Critical indicators (current price or RSI) are None. current_price: {current_price}, rsi: {rsi}. Cannot proceed with this stock.") # type: ignore
                 return None

            # Initialize all new signal values
            is_volume_breakout = False
            volume_ratio = None
            sma_20_50_crossed_bullish = None
            sma_50_200_crossed_bullish = None
            sma_20_50_crossed_bearish = None
            sma_50_200_crossed_bearish = None
            bb_breakout_upper = None
            bb_squeeze = None
            bb_upper = None
            bb_lower = None
            bb_ma = None
            # N-day High/Low breakout signals
            breakout_n_day_high_signal = None # Renamed to avoid conflict with dict key
            breakdown_n_day_low_signal = None # Renamed
            n_day_high_value = None           # Renamed
            n_day_low_value = None            # Renamed
            # MACD Crossover and RSI Recovery
            macd_bullish_crossover = None
            macd_bearish_crossover = None
            rsi_recovered_from_oversold = None
            current_macd_line = None
            current_macd_signal = None


            # SMA200 Calculation
            sma_200 = None
            min_len_for_sma200 = self.config.TechnicalThresholds.SMA_LONG
            if len(hist['Close']) >= min_len_for_sma200:
                sma_200_series = hist['Close'].rolling(window=min_len_for_sma200, min_periods=max(1, min_len_for_sma200 // 2)).mean()
                if not sma_200_series.empty:
                    sma_200 = sma_200_series.iloc[-1] # Get the last value
            else:
                logger.debug(f"Ticker '{ticker}': Not enough data points ({len(hist['Close'])}) for SMA200 calculation (period: {min_len_for_sma200}).")

            # MA Crossover Detections
            # Ensure we have at least 2 data points for SMAs to detect a crossover from previous to current
            if len(hist['Close']) >= 2: # General check for iloc[-2] access
                # Helper function for crossover detection
                def get_sma_series(window, min_p):
                    if len(hist['Close']) >= window:
                        return hist['Close'].rolling(window=window, min_periods=min_p).mean()
                    return None

                sma_20_series = get_sma_series(self.config.TechnicalThresholds.SMA_SHORT, max(1, self.config.TechnicalThresholds.SMA_SHORT // 2))
                sma_50_series = get_sma_series(self.config.TechnicalThresholds.SMA_MEDIUM, max(1, self.config.TechnicalThresholds.SMA_MEDIUM // 2))
                sma_200_series_for_crossover = get_sma_series(self.config.TechnicalThresholds.SMA_LONG, max(1, self.config.TechnicalThresholds.SMA_LONG//2))


                if sma_20_series is not None and sma_50_series is not None and len(sma_20_series) >=2 and len(sma_50_series) >=2:
                    if pd.notna(sma_20_series.iloc[-1]) and pd.notna(sma_50_series.iloc[-1]) and \
                       pd.notna(sma_20_series.iloc[-2]) and pd.notna(sma_50_series.iloc[-2]):
                        sma_20_50_crossed_bullish = sma_20_series.iloc[-1] > sma_50_series.iloc[-1] and \
                                                    sma_20_series.iloc[-2] <= sma_50_series.iloc[-2]
                        sma_20_50_crossed_bearish = sma_20_series.iloc[-1] < sma_50_series.iloc[-1] and \
                                                    sma_20_series.iloc[-2] >= sma_50_series.iloc[-2]

                if sma_50_series is not None and sma_200_series_for_crossover is not None and len(sma_50_series) >=2 and len(sma_200_series_for_crossover) >=2 :
                    if pd.notna(sma_50_series.iloc[-1]) and pd.notna(sma_200_series_for_crossover.iloc[-1]) and \
                       pd.notna(sma_50_series.iloc[-2]) and pd.notna(sma_200_series_for_crossover.iloc[-2]):
                        sma_50_200_crossed_bullish = sma_50_series.iloc[-1] > sma_200_series_for_crossover.iloc[-1] and \
                                                     sma_50_series.iloc[-2] <= sma_200_series_for_crossover.iloc[-2]
                        sma_50_200_crossed_bearish = sma_50_series.iloc[-1] < sma_200_series_for_crossover.iloc[-1] and \
                                                     sma_50_series.iloc[-2] >= sma_200_series_for_crossover.iloc[-2]

            # Bollinger Bands Calculation
            if ta: # Check if 'ta' library was imported successfully
                bb_config = self.config.BreakoutTechnicalThresholds
                if len(hist['Close']) >= bb_config.BB_WINDOW:
                    try:
                        indicator_bb = ta.volatility.BollingerBands(close=hist['Close'], window=bb_config.BB_WINDOW, window_dev=bb_config.BB_STD_DEV)
                        bb_ma_series = indicator_bb.bollinger_mavg()
                        bb_upper_series = indicator_bb.bollinger_hband()
                        bb_lower_series = indicator_bb.bollinger_lband()

                        if not all(s.empty for s in [bb_ma_series, bb_upper_series, bb_lower_series]):
                            bb_ma = bb_ma_series.iloc[-1]
                            bb_upper = bb_upper_series.iloc[-1]
                            bb_lower = bb_lower_series.iloc[-1]

                            if pd.notna(current_price) and pd.notna(bb_upper):
                                bb_breakout_upper = current_price > bb_upper

                            if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_ma) and bb_ma > 0: # bb_ma > 0 for squeeze calc
                                bb_squeeze = (bb_upper - bb_lower) < (bb_ma * bb_config.BB_SQUEEZE_STD_THRESHOLD)
                            else:
                                logger.debug(f"Ticker '{ticker}': BB values NaN, cannot calculate squeeze. MA:{bb_ma} Upper:{bb_upper} Lower:{bb_lower}")
                        else:
                            logger.warning(f"Ticker '{ticker}': Bollinger Band series are empty. Skipping BB calculations.")
                    except Exception as e_bb:
                        logger.error(f"Ticker '{ticker}': Error calculating Bollinger Bands: {e_bb}", exc_info=True)
                else:
                    logger.debug(f"Ticker '{ticker}': Not enough data points ({len(hist['Close'])}) for Bollinger Bands (window: {bb_config.BB_WINDOW}).")
            else:
                logger.warning(f"Ticker '{ticker}': Technical Analysis (ta) library not available. Skipping Bollinger Band calculations.")

            # N-day High/Low Breakout Calculation
            n_period_high_low = self.config.TechnicalThresholds.SMA_MEDIUM # Using 50-day period for N-day H/L

            if 'High' not in hist.columns or 'Low' not in hist.columns or hist['High'].empty or hist['Low'].empty:
                logger.warning(f"Ticker '{ticker}': 'High' or 'Low' columns not found or empty. Skipping N-day High/Low breakout.")
            elif len(hist['Close']) < n_period_high_low: # Need enough data for the N-day period
                logger.debug(f"Ticker '{ticker}': Not enough data points ({len(hist['Close'])}) for N-day High/Low (period: {n_period_high_low}).")
            else:
                # Calculate N-day high, excluding the current day's high for breakout context (compare current close to past N-day high)
                # So, use .iloc[:-1] to get all rows except the last, then roll.
                n_day_high_value = hist['High'].iloc[:-1].rolling(window=n_period_high_low, min_periods=max(1, n_period_high_low // 2)).max().iloc[-1]
                n_day_low_value = hist['Low'].iloc[:-1].rolling(window=n_period_high_low, min_periods=max(1, n_period_high_low // 2)).min().iloc[-1]

                if pd.notna(current_price) and pd.notna(n_day_high_value):
                    if current_price > n_day_high_value:
                        breakout_n_day_high_signal = True
                        logger.info(f"Ticker '{ticker}': Price broke above {n_period_high_low}-day high of {n_day_high_value:.2f}")
                    else:
                        breakout_n_day_high_signal = False # Explicitly false if not a breakout

                if pd.notna(current_price) and pd.notna(n_day_low_value):
                    if current_price < n_day_low_value:
                        breakdown_n_day_low_signal = True
                        logger.info(f"Ticker '{ticker}': Price broke below {n_period_high_low}-day low of {n_day_low_value:.2f}")
                    else:
                        breakdown_n_day_low_signal = False # Explicitly false

            # Ensure n_day_high_value and n_day_low_value are None if not calculated, for dict consistency
            if 'n_day_high_value' in locals() and pd.isna(n_day_high_value): n_day_high_value = None # Should be assigned None if not calculable
            if 'n_day_low_value' in locals() and pd.isna(n_day_low_value): n_day_low_value = None   # Should be assigned None if not calculable

            # MACD Calculation and Crossover Detection
            if ta:
                macd_cfg = self.config.BreakoutTechnicalThresholds
                min_len_for_macd = macd_cfg.MACD_SLOW_PERIOD + macd_cfg.MACD_SIGNAL_PERIOD # Approximate min length
                if len(hist['Close']) >= min_len_for_macd:
                    try:
                        macd_indicator = ta.trend.MACD(close=hist['Close'],
                                                       window_slow=macd_cfg.MACD_SLOW_PERIOD,
                                                       window_fast=macd_cfg.MACD_FAST_PERIOD,
                                                       window_sign=macd_cfg.MACD_SIGNAL_PERIOD)
                        macd_line_series = macd_indicator.macd()
                        macd_signal_series = macd_indicator.macd_signal()

                        if not macd_line_series.empty and not macd_signal_series.empty and \
                           len(macd_line_series) >= 2 and len(macd_signal_series) >= 2:
                            current_macd_line = macd_line_series.iloc[-1]
                            current_macd_signal = macd_signal_series.iloc[-1]

                            if pd.notna(current_macd_line) and pd.notna(current_macd_signal) and \
                               pd.notna(macd_line_series.iloc[-2]) and pd.notna(macd_signal_series.iloc[-2]):
                                macd_bullish_crossover = current_macd_line > current_macd_signal and \
                                                         macd_line_series.iloc[-2] <= macd_signal_series.iloc[-2]
                                macd_bearish_crossover = current_macd_line < current_macd_signal and \
                                                         macd_line_series.iloc[-2] >= macd_signal_series.iloc[-2]
                        else:
                            logger.debug(f"Ticker '{ticker}': MACD series too short or empty for crossover detection.")
                    except Exception as e_macd:
                        logger.error(f"Ticker '{ticker}': Error calculating MACD: {e_macd}", exc_info=True)
                else:
                    logger.debug(f"Ticker '{ticker}': Not enough data points ({len(hist['Close'])}) for MACD (needs approx {min_len_for_macd}).")
            else:
                logger.warning(f"Ticker '{ticker}': Technical Analysis (ta) library not available. Skipping MACD calculations.")

            # RSI Recovery from Oversold Detection
            # The existing `rsi` variable holds the latest RSI value. We need a series for recovery detection.
            if ta:
                # Standard RSI period is 14, already used for the single `rsi` value.
                # Let's ensure we have enough data for a small lookback for the recovery check.
                # The existing `rsi` is iloc[-1] of a 14-period rolling calculation on diffs.
                # For recovery, we need to see if it *was* oversold recently.
                # The current `rsi` value calculation is manual. For simplicity & consistency, let's use ta.momentum.RSIIndicator
                # if we need the series. The existing `rsi` is fine if we only check current rsi vs previous rsi.
                # However, the prompt asks for "any(rsi_series.iloc[-5:-1] < self.config.TechnicalThresholds.RSI_OVERSOLD))"
                # This implies needing an RSI series.

                rsi_cfg_recovery_lookback = 5 # How many previous periods to check for oversold state
                min_len_for_rsi_series = 14 + rsi_cfg_recovery_lookback # RSI period + lookback

                if len(hist['Close']) >= min_len_for_rsi_series:
                    try:
                        rsi_indicator_series_calc = ta.momentum.RSIIndicator(close=hist['Close'], window=14) # Default RSI window
                        rsi_series = rsi_indicator_series_calc.rsi()

                        if not rsi_series.empty and len(rsi_series) >= rsi_cfg_recovery_lookback : # Need at least lookback period entries
                            current_rsi_value_for_recovery = rsi_series.iloc[-1] # This should be very close to 'rsi' var

                            # Check if RSI was oversold in the recent past (e.g., previous 1 to 4 days)
                            # and is now above the recovery confirmation level.
                            # iloc[-5:-1] means indices from (length-5) up to (length-2), so -2, -3, -4, -5 relative to end.
                            # Ensure indices are valid if series is short, though len check above should help.
                            # A simple check: current > recovery_confirmation and previous <= oversold_threshold
                            # previous_rsi_value = rsi_series.iloc[-2]
                            # rsi_recovered_from_oversold = (pd.notna(current_rsi_value_for_recovery) and pd.notna(previous_rsi_value) and
                            #                               current_rsi_value_for_recovery > self.config.BreakoutTechnicalThresholds.RSI_RECOVERY_CONFIRMATION and
                            #                               previous_rsi_value <= self.config.TechnicalThresholds.RSI_OVERSOLD)

                            # Using the "any in last X days" logic from prompt:
                            if pd.notna(current_rsi_value_for_recovery):
                                if current_rsi_value_for_recovery > self.config.BreakoutTechnicalThresholds.RSI_RECOVERY_CONFIRMATION:
                                    # Ensure the slice [-rsi_cfg_recovery_lookback:-1] is valid
                                    # Check if any of the values in the slice (from index -lookback up to -2) were oversold
                                    # e.g., for lookback 5: indices -5, -4, -3, -2
                                    # Make sure series has enough points for this slice. len(rsi_series) must be >= lookback
                                    if len(rsi_series) > rsi_cfg_recovery_lookback: # e.g. if lookback is 5, need at least 6 items for series[-1] and series[-lookback-1] which is series[-6]
                                         # Correct slice would be from -(lookback_window + 1) up to -2 (exclusive of -1 which is current)
                                         # e.g., for lookback 5, check from hist.iloc[-6] to hist.iloc[-2]
                                        past_rsi_slice = rsi_series.iloc[-(rsi_cfg_recovery_lookback +1) : -1]
                                        if not past_rsi_slice.empty and any(past_rsi_slice < self.config.TechnicalThresholds.RSI_OVERSOLD):
                                            rsi_recovered_from_oversold = True
                                        else:
                                            rsi_recovered_from_oversold = False # Not recovered or wasn't oversold
                                    else: # Not enough historical RSI points for the lookback window.
                                        rsi_recovered_from_oversold = False # Cannot confirm recovery
                                else:
                                    rsi_recovered_from_oversold = False # Not above recovery threshold
                            else:
                                rsi_recovered_from_oversold = False # Current RSI is NaN
                        else:
                             logger.debug(f"Ticker '{ticker}': RSI series too short or empty for recovery detection.")
                             rsi_recovered_from_oversold = False
                    except Exception as e_rsi_rec:
                        logger.error(f"Ticker '{ticker}': Error calculating RSI series for recovery: {e_rsi_rec}", exc_info=True)
                        rsi_recovered_from_oversold = False
                else:
                    logger.debug(f"Ticker '{ticker}': Not enough data points ({len(hist['Close'])}) for RSI Recovery (needs approx {min_len_for_rsi_series}).")
                    rsi_recovered_from_oversold = False # Default if not enough data
            else:
                logger.warning(f"Ticker '{ticker}': Technical Analysis (ta) library not available. Skipping RSI Recovery calculations.")
                rsi_recovered_from_oversold = False


            # Volume Breakout Calculation (existing code from previous step)
            if 'Volume' not in hist.columns or hist['Volume'].empty:
                logger.warning(f"Ticker '{ticker}': 'Volume' column not found in history or is empty. Skipping volume breakout calculation.")
            else:
                vol_avg_period = self.config.BreakoutTechnicalThresholds.VOLUME_AVG_PERIOD
                # Ensure min_periods is at least 1 and not more than the window itself.
                min_periods_vol = max(1, min(vol_avg_period // 2, vol_avg_period))

                if len(hist['Volume']) >= min_periods_vol: # Check if there's enough data for a meaningful average
                    avg_volume = hist['Volume'].rolling(window=vol_avg_period, min_periods=min_periods_vol).mean().iloc[-1]
                    current_volume = hist['Volume'].iloc[-1]

                    if pd.isna(avg_volume) or avg_volume == 0:
                        logger.warning(f"Ticker '{ticker}': Average volume is NaN or zero (avg_volume: {avg_volume}). Volume ratio cannot be calculated.")
                        # volume_ratio remains None
                    elif pd.isna(current_volume):
                        logger.warning(f"Ticker '{ticker}': Current volume is NaN. Volume ratio cannot be calculated.")
                        # volume_ratio remains None
                    else:
                        volume_ratio = current_volume / avg_volume
                        if volume_ratio >= self.config.BreakoutTechnicalThresholds.VOLUME_SURGE_MULTIPLIER:
                            is_volume_breakout = True
                            logger.info(f"Ticker '{ticker}': Volume breakout detected! Ratio: {volume_ratio:.2f}")
                        else:
                            logger.debug(f"Ticker '{ticker}': No volume breakout. Ratio: {volume_ratio:.2f}")
                else:
                    logger.warning(f"Ticker '{ticker}': Not enough data points ({len(hist['Volume'])}) for average volume calculation (period: {vol_avg_period}, min_periods: {min_periods_vol}).")


            return_dict = {
                'ticker': ticker, # Normalized ticker
                'company_name': info.get('longName', original_ticker), # Use original ticker for company name if longName is missing
                'current_price': current_price,
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                # Use current_volume if calculated, otherwise default from hist (which might be 0 if column missing)
                'volume': current_volume if 'current_volume' in locals() and pd.notna(current_volume) else (hist['Volume'].iloc[-1] if 'Volume' in hist.columns and not hist['Volume'].empty else 0),
                'rsi': rsi,
                'sma_20': sma_20,
                'sma_50': sma_50 if sma_50 is not None else sma_20, # Fallback sma_50 to sma_20 if it was not calculable
                'momentum_20d': momentum_20d,
                'pe_ratio': info.get('trailingPE'), # Allow None for PE
                'roe': info.get('returnOnEquity'), # Allow None
                'debt_to_equity': info.get('debtToEquity'), # Allow None
                'profit_margin': info.get('profitMargins'), # Allow None
                # New breakout fields
                'is_volume_breakout': is_volume_breakout,
                'volume_ratio': volume_ratio,
                # MA Crossover signals
                'sma_20_50_crossed_bullish': sma_20_50_crossed_bullish,
                'sma_50_200_crossed_bullish': sma_50_200_crossed_bullish,
                'sma_20_50_crossed_bearish': sma_20_50_crossed_bearish,
                'sma_50_200_crossed_bearish': sma_50_200_crossed_bearish,
                # Bollinger Band signals
                'bb_breakout_upper': bb_breakout_upper,
                'bb_squeeze': bb_squeeze,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_ma': bb_ma,
                # Store SMA200 for potential use in scoring or other signals
                'sma_200': sma_200 if pd.notna(sma_200) else None,
                # N-day High/Low breakout signals
                'breakout_n_day_high': breakout_n_day_high_signal,
                'breakdown_n_day_low': breakdown_n_day_low_signal,
                'n_day_high': n_day_high_value,
                'n_day_low': n_day_low_value,
                # MACD and RSI Recovery
                'macd_bullish_crossover': macd_bullish_crossover,
                'macd_bearish_crossover': macd_bearish_crossover,
                'rsi_recovered_from_oversold': rsi_recovered_from_oversold,
                'macd_line': current_macd_line if pd.notna(current_macd_line) else None,
                'macd_signal_line': current_macd_signal if pd.notna(current_macd_signal) else None,
                'hist_data_df': hist.copy() if hist is not None else None,
            }
            return return_dict
        except Exception as e:
            logger.error(f"Error processing financial data for {ticker} after fetching: {e}", exc_info=True)
            return None
    
    def calculate_technical_score(self, data: Dict) -> Tuple[float, List[str]]:
        score = 50.0  # Start with a neutral score
        reasons = []

        if not data:
            return 0.0, ["Input data is missing for technical score"]

        ticker = data.get('ticker', 'Unknown') # For logging

        # Helper to check for None or NaN
        def is_valid_indicator(value):
            if value is None:
                return False
            if np and isinstance(value, float) and np.isnan(value):
                return False
            return True

        current_price = data.get('current_price')
        rsi_val = data.get('rsi')
        sma_20_val = data.get('sma_20')
        sma_50_val = data.get('sma_50')
        momentum_val = data.get('momentum_20d')

        # RSI Scoring
        if is_valid_indicator(rsi_val):
            if rsi_val < self.config.TechnicalThresholds.RSI_OVERSOLD: # type: ignore
                score += 15
                reasons.append(f"RSI oversold ({rsi_val:.1f})") # type: ignore
            elif rsi_val > self.config.TechnicalThresholds.RSI_OVERBOUGHT: # type: ignore
                score -= 15
                reasons.append(f"RSI overbought ({rsi_val:.1f})") # type: ignore
            else:
                reasons.append(f"RSI neutral ({rsi_val:.1f})") # type: ignore
        else:
            reasons.append("RSI: N/A or invalid")
            logger.debug(f"RSI data not available or invalid for {ticker}")


        # Moving Average Scoring
        # Only score SMAs if all relevant SMAs and current_price are valid
        if is_valid_indicator(current_price) and is_valid_indicator(sma_20_val) and is_valid_indicator(sma_50_val):
            if current_price > sma_20_val > sma_50_val: # type: ignore
                score += 10
                reasons.append(f"Price above key moving averages (Price: {current_price:.2f}, SMA20: {sma_20_val:.2f}, SMA50: {sma_50_val:.2f})") # type: ignore
            elif current_price < sma_20_val < sma_50_val: # type: ignore
                score -= 10
                reasons.append(f"Price below key moving averages (Price: {current_price:.2f}, SMA20: {sma_20_val:.2f}, SMA50: {sma_50_val:.2f})") # type: ignore
            else:
                reasons.append(f"Price relative to MAs is neutral/mixed (Price: {current_price:.2f}, SMA20: {sma_20_val:.2f}, SMA50: {sma_50_val:.2f})") # type: ignore
        elif is_valid_indicator(current_price) and is_valid_indicator(sma_20_val): # Check for SMA20 if SMA50 is not available
            if current_price > sma_20_val: # type: ignore
                score += 5 # Less points if only SMA20 is available
                reasons.append(f"Price above SMA20 (Price: {current_price:.2f}, SMA20: {sma_20_val:.2f})") # type: ignore
            elif current_price < sma_20_val: # type: ignore
                score -= 5
                reasons.append(f"Price below SMA20 (Price: {current_price:.2f}, SMA20: {sma_20_val:.2f})") # type: ignore
        else:
            reasons.append("Moving Averages: N/A or insufficient data (Price, SMA20, or SMA50)")
            logger.debug(f"SMA data not available or invalid for {ticker}. Price: {current_price}, SMA20: {sma_20_val}, SMA50: {sma_50_val}")

        # Momentum Scoring
        if is_valid_indicator(momentum_val):
            if momentum_val > 5: # type: ignore
                score += 10
                reasons.append(f"Strong positive momentum ({momentum_val:.1f}%)") # type: ignore
            elif momentum_val < -5: # type: ignore
                score -= 10
                reasons.append(f"Negative momentum ({momentum_val:.1f}%)") # type: ignore
            else:
                reasons.append(f"Momentum neutral ({momentum_val:.1f}%)") # type: ignore
        else:
            reasons.append("Momentum (20d): N/A or invalid")
            logger.debug(f"Momentum data not available or invalid for {ticker}")
        
        # Ensure current_price is valid before final score adjustment based on it.
        # This part of original code was problematic as it assumed all indicators were valid.
        # The previous block `if None in [current_price, rsi, sma_20, sma_50, momentum]:`
        # would return 0.0, which is not ideal if only one indicator is missing.
        # The new approach scores based on available data.
        # The initial check `if not data:` handles the case of no data at all.
        # If current_price itself is None, most other things would be too, or calculations would be skewed.
        if not is_valid_indicator(current_price):
            logger.warning(f"Current price is None or NaN for {ticker}. Technical score might be unreliable or zero.")
            # Depending on strictness, could return 0 here.
            # For now, allow score to be based on other valid indicators.
            # If no indicators were valid, score would remain 50 (neutral).
            # If strict, and current_price is mandatory:
            # return 0.0, ["Current price is N/A, cannot calculate technical score"]
        
        return max(0, min(100, score)), reasons

    def calculate_breakout_score(self, data: Dict) -> Tuple[Optional[float], List[str]]:
        score = 0.0  # Initialize to 0, as it's an additive score based on positive signals
        reasons = []
        ticker = data.get('ticker', 'Unknown') # For logging

        if not data:
            logger.warning(f"Ticker '{ticker}': No data provided for breakout scoring.")
            return None, ["No data for breakout scoring"]

        weights = self.config.BreakoutScoringWeights

        # Helper to safely get boolean signals, defaulting to False if None or key missing
        def get_signal(key: str) -> bool:
            val = data.get(key)
            return val if isinstance(val, bool) else False

        is_volume_breakout = get_signal('is_volume_breakout')
        breakout_n_day_high = get_signal('breakout_n_day_high')
        sma_20_50_crossed_bullish = get_signal('sma_20_50_crossed_bullish')
        sma_50_200_crossed_bullish = get_signal('sma_50_200_crossed_bullish')
        bb_breakout_upper = get_signal('bb_breakout_upper')
        bb_squeeze = get_signal('bb_squeeze')
        macd_bullish_crossover = get_signal('macd_bullish_crossover')
        rsi_recovered_from_oversold = get_signal('rsi_recovered_from_oversold')

        if is_volume_breakout:
            score += weights.VOLUME_BREAKOUT_WEIGHT * 100
            reasons.append(f"Volume breakout (ratio: {data.get('volume_ratio', 0.0):.2f}x avg)")
            logger.debug(f"Ticker '{ticker}': +{weights.VOLUME_BREAKOUT_WEIGHT*100} for volume breakout.")

        if breakout_n_day_high:
            n_day_high_period = self.config.TechnicalThresholds.SMA_MEDIUM # Assuming this was the period used
            score += weights.PRICE_BREAKOUT_S_R_WEIGHT * 100
            reasons.append(f"Price broke {n_day_high_period}-day high ({data.get('n_day_high', 0.0):.2f})")
            logger.debug(f"Ticker '{ticker}': +{weights.PRICE_BREAKOUT_S_R_WEIGHT*100} for N-day high breakout.")

        # MA Crossovers - prioritize stronger one
        ma_crossover_scored = False
        if sma_50_200_crossed_bullish:
            score += weights.MA_CROSSOVER_WEIGHT * 100
            reasons.append("Bullish MA Crossover (SMA50 > SMA200)")
            logger.debug(f"Ticker '{ticker}': +{weights.MA_CROSSOVER_WEIGHT*100} for SMA50/200 crossover.")
            ma_crossover_scored = True
        elif sma_20_50_crossed_bullish and not ma_crossover_scored: # Only score if the stronger one hasn't occurred
            score += weights.MA_CROSSOVER_WEIGHT * 0.75 * 100 # Apply a factor for the less strong signal
            reasons.append("Bullish MA Crossover (SMA20 > SMA50)")
            logger.debug(f"Ticker '{ticker}': +{weights.MA_CROSSOVER_WEIGHT*0.75*100} for SMA20/50 crossover.")

        if bb_breakout_upper:
            score += weights.BB_BREAKOUT_WEIGHT * 100
            reasons.append(f"Price broke upper Bollinger Band ({data.get('bb_upper', 0.0):.2f})")
            logger.debug(f"Ticker '{ticker}': +{weights.BB_BREAKOUT_WEIGHT*100} for BB upper breakout.")

        if bb_squeeze:
            score += weights.BB_SQUEEZE_WEIGHT * 100
            reasons.append("Bollinger Bands Squeeze (potential energy)")
            logger.debug(f"Ticker '{ticker}': +{weights.BB_SQUEEZE_WEIGHT*100} for BB squeeze.")
            # Bonus for breakout from squeeze
            if bb_breakout_upper:
                bonus_squeeze_breakout = 0.05 * 100 # Example bonus: 5 points
                score += bonus_squeeze_breakout
                reasons.append("Bonus: Breakout from BB Squeeze")
                logger.debug(f"Ticker '{ticker}': +{bonus_squeeze_breakout} bonus for BB squeeze breakout.")


        if macd_bullish_crossover:
            score += weights.MACD_CROSSOVER_WEIGHT * 100
            reasons.append("MACD bullish crossover")
            logger.debug(f"Ticker '{ticker}': +{weights.MACD_CROSSOVER_WEIGHT*100} for MACD crossover.")

        if rsi_recovered_from_oversold:
            score += weights.RSI_RECOVERY_WEIGHT * 100
            reasons.append(f"RSI recovered from oversold (RSI: {data.get('rsi', 0.0):.1f})")
            logger.debug(f"Ticker '{ticker}': +{weights.RSI_RECOVERY_WEIGHT*100} for RSI recovery.")

        if not reasons: # No breakout signals triggered
            logger.info(f"Ticker '{ticker}': No breakout signals detected for breakout score.")
            return 0.0, ["No specific breakout signals detected."] # Return 0 score and a neutral reason

        return max(0, min(100, score)), reasons
    
    def calculate_fundamental_score(self, data: Dict) -> Tuple[float, List[str]]:
        score = 50.0
        reasons = []

        if not data:
            return 0, ["Input data is missing for fundamental score"]
        
        pe_ratio = data.get('pe_ratio')
        roe = data.get('roe')
        debt_to_equity = data.get('debt_to_equity')

        if None in [pe_ratio, roe, debt_to_equity]:
            logger.warning(f"Missing one or more data points for fundamental score calculation for {data.get('ticker', 'Unknown')}.")
            reasons.append("Missing critical data for fundamental scoring")
            return 0.0, reasons # Or 50.0 if neutral is preferred
        
        if 0 < pe_ratio < self.config.FundamentalThresholds.PE_EXCELLENT_MAX: # type: ignore
            score += 15
            reasons.append(f"Excellent P/E ratio ({pe_ratio:.1f})") # type: ignore
        elif pe_ratio > self.config.FundamentalThresholds.PE_POOR_MIN: # type: ignore
            score -= 10
            reasons.append(f"High P/E ratio ({pe_ratio:.1f})")
        
        if roe > self.config.FundamentalThresholds.ROE_EXCELLENT_MIN:
            score += 15
            reasons.append(f"Excellent ROE ({roe:.1%})")
        elif roe > self.config.FundamentalThresholds.ROE_GOOD_MIN:
            score += 8
            reasons.append(f"Good ROE ({roe:.1%})")
        elif roe < 0:
            score -= 15
            reasons.append("Negative ROE")
        
        if debt_to_equity < 0.3:
            score += 8
            reasons.append("Low debt levels")
        elif debt_to_equity > 1.0:
            score -= 8
            reasons.append("High debt levels")
        
        return max(0, min(100, score)), reasons
      def analyze_stock(self, ticker: str) -> Optional[StockAnalysis]:
        # Note: ticker here is the original, un-normalized ticker from the input list
        logger.info(f"Starting analysis for ticker: '{ticker}'...")
        try:
            data = self.get_stock_data(ticker) # get_stock_data will normalize it
            if not data:
                # get_stock_data already logs detailed reasons for failure.
                # Add a specific message for skipping analysis.
                logger.warning(f"Analysis skipped for ticker '{ticker}': No data obtained or ticker invalid.")
                return None
            
            # data['ticker'] will be the normalized ticker
            normalized_ticker = data['ticker']
            logger.info(f"Data obtained for '{normalized_ticker}'. Calculating scores...")
            technical_score, technical_reasons = self.calculate_technical_score(data)
            fundamental_score, fundamental_reasons = self.calculate_fundamental_score(data)
            breakout_score, breakout_reasons = self.calculate_breakout_score(data)
            
            # Retrieve hist_data_df from data dictionary to pass to calculate_trading_parameters
            hist_df_for_trading_params = data.get('hist_data_df')
            trading_parameters = self.calculate_trading_parameters(data, hist_df_for_trading_params)


            overall_score = ( # Current overall_score remains unchanged, breakout_score is separate
                technical_score * self.config.ScoringWeights.TECHNICAL_WEIGHT +
                fundamental_score * self.config.ScoringWeights.FUNDAMENTAL_WEIGHT +
                50 * self.config.ScoringWeights.SENTIMENT_WEIGHT
            )
            
            if overall_score >= self.config.ScoringWeights.STRONG_BUY_THRESHOLD:
                recommendation = "STRONG_BUY"
                confidence = 0.9
            elif overall_score >= self.config.ScoringWeights.BUY_THRESHOLD:
                recommendation = "BUY"
                confidence = 0.7
            elif overall_score >= self.config.ScoringWeights.HOLD_THRESHOLD:
                recommendation = "HOLD"
                confidence = 0.5
            elif overall_score >= self.config.ScoringWeights.SELL_THRESHOLD:
                recommendation = "SELL"
                confidence = 0.7
            else:
                recommendation = "STRONG_SELL"
                confidence = 0.9
            
            all_reasons = technical_reasons + fundamental_reasons
            
            return StockAnalysis(
                ticker=normalized_ticker, # Use the normalized ticker from data
                company_name=data['company_name'],
                current_price=data['current_price'],
                sector=data['sector'],
                rsi=data['rsi'],
                sma_20=data['sma_20'],
                sma_50=data['sma_50'],
                momentum_20d=data['momentum_20d'],
                pe_ratio=data['pe_ratio'],
                roe=data['roe'],
                debt_to_equity=data['debt_to_equity'],
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                overall_score=overall_score,
                breakout_score=breakout_score,
                recommendation=recommendation,
                confidence=confidence,
                reasons=all_reasons,
                breakout_reasons=breakout_reasons,
                # Add new trading parameters
                suggested_entry_price=trading_parameters.get('suggested_entry_price'),
                target_price_1=trading_parameters.get('target_price_1'),
                stop_loss_price=trading_parameters.get('stop_loss_price'),
                risk_reward_ratio=trading_parameters.get('risk_reward_ratio'),
                risk_level=trading_parameters.get('risk_level'),
                max_position_size_pct=trading_parameters.get('max_position_size_pct'),
                trading_strategy_summary=trading_parameters.get('trading_strategy_summary'),
                time_horizon=trading_parameters.get('time_horizon')
            )
        except Exception as e:
            logger.error(f"Unexpected error during detailed analysis of ticker '{data.get('ticker', ticker)}': {e}", exc_info=True) # Use normalized ticker if available
            return None
    
    def analyze_multiple_stocks(self, tickers: List[str]) -> List[StockAnalysis]:
        if not tickers:
            logger.warning("No tickers provided for analysis.")
            return []
        logger.info(f"Starting analysis for {len(tickers)} stock(s): {tickers}")
        results = []
        for i, ticker_input in enumerate(tickers, 1):
            # Ticker input here is exactly as provided in the list
            logger.info(f"Processing {i}/{len(tickers)}: '{ticker_input}'")
            analysis = self.analyze_stock(ticker_input)
            if analysis:
                results.append(analysis)
            # analyze_stock and get_stock_data handle their own detailed logging for failures
        logger.info(f"Analysis completed. {len(results)} stocks analyzed successfully.")
        return results
    
    def get_recommendations(self, analyses: List[StockAnalysis], 
                          min_score: float = 60) -> Tuple[List[StockAnalysis], List[StockAnalysis]]:
        buy_recommendations = [
            analysis for analysis in analyses 
            if analysis.recommendation in ["STRONG_BUY", "BUY"] and analysis.overall_score >= min_score
        ]
        sell_recommendations = [
            analysis for analysis in analyses 
            if analysis.recommendation in ["STRONG_SELL", "SELL"] and analysis.overall_score <= (100 - min_score)
        ]
        buy_recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        sell_recommendations.sort(key=lambda x: x.overall_score)
        return buy_recommendations, sell_recommendations

def main():
    analyzer = StockAnalyzer()

    # Check if critical modules loaded, pd, yf are most critical for fetching and processing.
    # np is often a dependency of pandas or yfinance, so covered by them.
    # requests is used by yfinance internally or if we add direct web calls.
    if not pd or not yf:
        logger.critical("Critical libraries (Pandas or yfinance) not loaded. Aborting main execution.")
        print("❌ Critical libraries (Pandas or yfinance) failed to load. Please check the log and install missing packages. Exiting.")
        return

    # Updated test tickers for breakout system testing
    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'JNJ', 'FSLY', 'BADTICKER']

    print(f"🧪 Starting breakout system test analysis for tickers: {test_tickers}")
    
    # analyze_multiple_stocks returns a list of StockAnalysis objects for successfully analyzed stocks
    analyzed_results = analyzer.analyze_multiple_stocks(test_tickers)

    successfully_analyzed_tickers = {result.ticker for result in analyzed_results} # Normalized tickers

    print("\n📋 Test Analysis Summary:")
    print("-" * 30)

    for ticker_input in test_tickers:
        # Need to find if the original ticker_input (or its normalized form) was analyzed
        # get_stock_data normalizes, so data.ticker in StockAnalysis object is normalized.
        # We compare normalized versions for robustness.
        normalized_input = ticker_input.strip().upper()
        found_analysis = None
        for analysis in analyzed_results:
            if analysis.ticker == normalized_input: # analysis.ticker is already normalized
                found_analysis = analysis
                break

        if found_analysis:
            print(f"\n✅ {found_analysis.ticker} ({found_analysis.company_name}):")
            print(f"  Price: ${found_analysis.current_price:.2f}")
            print(f"  Price: ${found_analysis.current_price:.2f}, RSI: {found_analysis.rsi:.2f if found_analysis.rsi is not None else 'N/A'}") # type: ignore
            print(f"  Overall Score: {found_analysis.overall_score:.1f}/100, Recommendation: {found_analysis.recommendation} ({found_analysis.confidence:.0%})") # type: ignore
            if found_analysis.reasons: # type: ignore
                print(f"  Std. Reasons: {'; '.join(found_analysis.reasons)}") # type: ignore

            print(f"  --- Breakout Signals & Score ---")
            # Individual signals are now part of the trading plan summary or can be logged for debug
            if found_analysis.breakout_score is not None: # type: ignore
                print(f"    Breakout Score: {found_analysis.breakout_score:.1f}/100") # type: ignore
            if found_analysis.breakout_reasons: # type: ignore
                print(f"    Breakout Reasons: {'; '.join(found_analysis.breakout_reasons)}") # type: ignore

            print(f"  --- Trading Plan ({found_analysis.risk_level}) ---") # type: ignore
            if found_analysis.trading_strategy_summary: # type: ignore
                # More detailed print for each component of the trading plan
                print(f"    Summary: {found_analysis.trading_strategy_summary}") # type: ignore
                if found_analysis.suggested_entry_price is not None:print(f"      Suggested Entry: ~${found_analysis.suggested_entry_price:.2f}") # type: ignore
                if found_analysis.stop_loss_price is not None: print(f"      Stop-Loss: ~${found_analysis.stop_loss_price:.2f}") # type: ignore
                if found_analysis.target_price_1 is not None: print(f"      Target 1: ~${found_analysis.target_price_1:.2f}") # type: ignore
                if found_analysis.risk_reward_ratio is not None: print(f"      Risk/Reward Ratio: {found_analysis.risk_reward_ratio:.2f}:1") # type: ignore
                if found_analysis.max_position_size_pct is not None: print(f"      Max Position Size: {found_analysis.max_position_size_pct:.0%}") # type: ignore
            else:
                print("    No trading plan generated (e.g., breakout score too low or data missing).")
        else:
            # This ticker_input was not successfully analyzed. Logs should indicate why.
            print(f"\n❌ {ticker_input.upper()}: Analysis failed or skipped (see logs for details).")

    print("-" * 30)
    print(f"📈 Test analysis completed. Successfully analyzed {len(analyzed_results)} out of {len(test_tickers)} tickers.")

if __name__ == "__main__":
    # Setup logger to show DEBUG messages for this test run if desired
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
