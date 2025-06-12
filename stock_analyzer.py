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
    rsi: float # Optional[float] might be better if it can be None
    sma_20: float # Optional[float]
    sma_50: float # Optional[float]
    momentum_20d: float # Optional[float]
    pe_ratio: float # Optional[float]
    roe: float # Optional[float]
    debt_to_equity: float # Optional[float]
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
    risk_level: Optional[str] = None
    max_position_size_pct: Optional[float] = None
    trading_strategy_summary: Optional[str] = None
    time_horizon: Optional[str] = None


class StockAnalyzer:
    def __init__(self):
        self.config = Config()
        if pd and np and yf and requests and ta:
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
            'trading_strategy_summary': "Strategy not determined.",
            'current_atr': None,
            'time_horizon': None
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
            trading_params['risk_level'] = risk_level_str
            trading_params['trading_strategy_summary'] = f"No actionable trade setup: Breakout score {breakout_score_val:.1f} too low."
            return trading_params

        trading_params['risk_level'] = risk_level_str
        trading_params['max_position_size_pct'] = max_pos_pct
        trading_params['time_horizon'] = cfg_trading.DEFAULT_TRADE_TIME_HORIZON

        suggested_entry = current_price
        trading_params['suggested_entry_price'] = suggested_entry

        if stop_loss_pct is not None and suggested_entry is not None:
            trading_params['stop_loss_price'] = suggested_entry * (1 - stop_loss_pct)

        if target_profit_pct is not None and suggested_entry is not None:
            trading_params['target_price_1'] = suggested_entry * (1 + target_profit_pct)

        sl_price = trading_params['stop_loss_price']
        tp_price = trading_params['target_price_1']

        rr_ratio_val = None
        if suggested_entry is not None and isinstance(sl_price, (float, int)) and isinstance(tp_price, (float, int)):
            potential_reward = tp_price - suggested_entry
            potential_risk = suggested_entry - sl_price
            if potential_risk > 0.0001:
                rr_ratio_val = potential_reward / potential_risk
                trading_params['risk_reward_ratio'] = rr_ratio_val
            else:
                logger.warning(f"Ticker '{ticker}': Potential risk is zero or negative. R/R not calculated. Entry: {suggested_entry}, SL: {sl_price}")

        current_atr_val = None
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
                         current_atr_val = calculated_atr
                         logger.debug(f"Ticker '{ticker}': Calculated ATR({cfg_trading.ATR_PERIOD}) = {current_atr_val:.2f}")
                else:
                    logger.debug(f"Ticker '{ticker}': Not enough data for ATR({cfg_trading.ATR_PERIOD}) calculation.")
            except Exception as e_atr:
                logger.error(f"Ticker '{ticker}': Error calculating ATR: {e_atr}", exc_info=True)

        summary = f"Strategy: {risk_level_str}"
        if suggested_entry is not None: summary += f" | Entry: Market ~${suggested_entry:.2f} on breakout confirmation"
        if sl_price is not None: summary += f" | SL: ~${sl_price:.2f}"
        if tp_price is not None: summary += f" | TP1: ~${tp_price:.2f}"
        if rr_ratio_val is not None: summary += f" | R/R: {rr_ratio_val:.2f}:1"
        if max_pos_pct is not None: summary += f" | Max Pos Size: {max_pos_pct:.0%}"
        if trading_params['time_horizon'] is not None: summary += f" | Horizon: {trading_params['time_horizon']}"

        trading_params['trading_strategy_summary'] = summary
        logger.info(f"Ticker '{ticker}': Trading parameters. Summary: {summary}")

        return trading_params

    def get_sp500_tickers(self) -> List[str]:
        try:
            if not pd:
                logger.error("Pandas is not available, cannot retrieve S&P 500 tickers. Using fallback list.")
                return self.config.FALLBACK_TICKERS
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving S&P 500 list: {e}. Using fallback list.")
            return self.config.FALLBACK_TICKERS

    def get_stock_data(self, ticker: str) -> Optional[Dict]:
        if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
            logger.warning(f"Invalid ticker format: Ticker is None, not a string, or empty. Value: '{ticker}'")
            return None

        original_ticker = ticker
        ticker = ticker.strip().upper()

        if len(ticker) > 10:
            logger.warning(f"Invalid ticker format: Ticker '{ticker}' (Original: '{original_ticker}') seems too long.")
            return None

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
                info = stock.info

                if not info or info.get('regularMarketPrice') is None or info.get('financialCurrency') is None :
                    logger.warning(f"Ticker '{ticker}': No valid detailed info (e.g., market price, currency) found on attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}. Ticker might be delisted, invalid, or not a standard stock type.")
                    if attempt == self.config.RETRY_ATTEMPTS - 1:
                        logger.error(f"Ticker '{ticker}': Failed to get valid detailed info after {self.config.RETRY_ATTEMPTS} attempts. Aborting for this ticker.")
                        return None

                hist = stock.history(period="3mo", timeout=self.config.API_TIMEOUT)

                if hist is not None and not hist.empty:
                    if len(hist) >= self.config.MIN_DATA_POINTS:
                        logger.info(f"Ticker '{ticker}': Successfully fetched history data on attempt {attempt + 1}.")
                        break
                    else:
                        logger.warning(f"Ticker '{ticker}': Insufficient data points ({len(hist)}) on attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}. Need {self.config.MIN_DATA_POINTS}.")
                else:
                    logger.warning(f"Ticker '{ticker}': No history data returned on attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}.")

                if attempt < self.config.RETRY_ATTEMPTS - 1:
                    logger.info(f"Ticker '{ticker}': Retrying...")
                    time.sleep(self.config.RETRY_DELAY_SECONDS)

            except Exception as e:
                last_exception = e
                logger.warning(f"Ticker '{ticker}': Error during data fetching (attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}): {e}. Retrying...")
                if attempt < self.config.RETRY_ATTEMPTS - 1:
                    time.sleep(self.config.RETRY_DELAY_SECONDS)

        if hist is None or hist.empty or len(hist) < self.config.MIN_DATA_POINTS:
            logger.error(f"Ticker '{ticker}': Failed to fetch sufficient history after {self.config.RETRY_ATTEMPTS} attempts. Min points: {self.config.MIN_DATA_POINTS}, Got: {len(hist) if hist is not None else 0}. Last error: {last_exception}")
            return None

        if not info or info.get('regularMarketPrice') is None or info.get('financialCurrency') is None:
             logger.error(f"Ticker '{ticker}': Valid detailed info could not be confirmed even if history was available. Cannot proceed.")
             return None

        try:
            if 'Close' not in hist.columns or hist['Close'].empty:
                logger.warning(f"Ticker '{ticker}': No 'Close' price data in history.")
                return None
            
            current_price = hist['Close'].iloc[-1]

            min_len_for_sma20 = 20
            min_len_for_sma50 = 50
            min_len_for_rsi = 15
            min_len_for_momentum = 21

            sma_20 = hist['Close'].rolling(window=min_len_for_sma20).mean().iloc[-1] if len(hist['Close']) >= min_len_for_sma20 else None
            sma_50 = hist['Close'].rolling(window=min_len_for_sma50).mean().iloc[-1] if len(hist['Close']) >= min_len_for_sma50 else (sma_20 if sma_20 is not None else None)

            rsi_val = None # Renamed from rsi to avoid conflict with the rsi_series variable later
            if len(hist['Close']) >= min_len_for_rsi:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean().iloc[-1]

                if np and (np.isnan(gain) or np.isnan(loss)):
                    rsi_val = np.nan
                elif loss == 0:
                    if gain == 0:
                        rsi_val = 50.0
                    else:
                        rsi_val = 100.0
                elif gain == 0:
                    rsi_val = 0.0
                else:
                    rs_val = gain / loss # Renamed from rs
                    if np and (np.isnan(rs_val) or np.isinf(rs_val)):
                        rsi_val = np.nan
                    else:
                        rsi_val = 100.0 - (100.0 / (1.0 + rs_val))
            else:
                rsi_val = None

            momentum_20d = ((current_price / hist['Close'].iloc[-min_len_for_momentum]) - 1) * 100 if len(hist['Close']) >= min_len_for_momentum and current_price is not None else None

            if np and isinstance(rsi_val, float) and np.isnan(rsi_val):
                rsi_val = None

            if current_price is None or rsi_val is None:
                 logger.error(f"Ticker '{ticker}': Critical indicators (current price or RSI) are None. current_price: {current_price}, rsi: {rsi_val}. Cannot proceed with this stock.")
                 return None

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
            breakout_n_day_high_signal = None
            breakdown_n_day_low_signal = None
            n_day_high_value = None
            n_day_low_value = None
            macd_bullish_crossover = None
            macd_bearish_crossover = None
            rsi_recovered_from_oversold = None
            current_macd_line = None
            current_macd_signal = None

            sma_200 = None
            min_len_for_sma200 = self.config.TechnicalThresholds.SMA_LONG
            if len(hist['Close']) >= min_len_for_sma200:
                sma_200_series_val = hist['Close'].rolling(window=min_len_for_sma200, min_periods=max(1, min_len_for_sma200 // 2)).mean() # Renamed
                if not sma_200_series_val.empty:
                    sma_200 = sma_200_series_val.iloc[-1]
            else:
                logger.debug(f"Ticker '{ticker}': Not enough data points ({len(hist['Close'])}) for SMA200 calculation (period: {min_len_for_sma200}).")

            if len(hist['Close']) >= 2:
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

            if ta:
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

                            if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_ma) and bb_ma > 0:
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

            n_period_high_low = self.config.TechnicalThresholds.SMA_MEDIUM

            if 'High' not in hist.columns or 'Low' not in hist.columns or hist['High'].empty or hist['Low'].empty:
                logger.warning(f"Ticker '{ticker}': 'High' or 'Low' columns not found or empty. Skipping N-day High/Low breakout.")
            elif len(hist['Close']) < n_period_high_low:
                logger.debug(f"Ticker '{ticker}': Not enough data points ({len(hist['Close'])}) for N-day High/Low (period: {n_period_high_low}).")
            else:
                n_day_high_value = hist['High'].iloc[:-1].rolling(window=n_period_high_low, min_periods=max(1, n_period_high_low // 2)).max().iloc[-1]
                n_day_low_value = hist['Low'].iloc[:-1].rolling(window=n_period_high_low, min_periods=max(1, n_period_high_low // 2)).min().iloc[-1]

                if pd.notna(current_price) and pd.notna(n_day_high_value):
                    if current_price > n_day_high_value:
                        breakout_n_day_high_signal = True
                        logger.info(f"Ticker '{ticker}': Price broke above {n_period_high_low}-day high of {n_day_high_value:.2f}")
                    else:
                        breakout_n_day_high_signal = False

                if pd.notna(current_price) and pd.notna(n_day_low_value):
                    if current_price < n_day_low_value:
                        breakdown_n_day_low_signal = True
                        logger.info(f"Ticker '{ticker}': Price broke below {n_period_high_low}-day low of {n_day_low_value:.2f}")
                    else:
                        breakdown_n_day_low_signal = False

            if 'n_day_high_value' in locals() and pd.isna(n_day_high_value): n_day_high_value = None
            if 'n_day_low_value' in locals() and pd.isna(n_day_low_value): n_day_low_value = None

            if ta:
                macd_cfg = self.config.BreakoutTechnicalThresholds
                min_len_for_macd = macd_cfg.MACD_SLOW_PERIOD + macd_cfg.MACD_SIGNAL_PERIOD
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

            if ta:
                rsi_cfg_recovery_lookback = 5
                min_len_for_rsi_series = 14 + rsi_cfg_recovery_lookback

                if len(hist['Close']) >= min_len_for_rsi_series:
                    try:
                        rsi_indicator_series_calc = ta.momentum.RSIIndicator(close=hist['Close'], window=14)
                        rsi_series_val = rsi_indicator_series_calc.rsi() # Renamed

                        if not rsi_series_val.empty and len(rsi_series_val) >= rsi_cfg_recovery_lookback :
                            current_rsi_value_for_recovery = rsi_series_val.iloc[-1]

                            if pd.notna(current_rsi_value_for_recovery):
                                if current_rsi_value_for_recovery > self.config.BreakoutTechnicalThresholds.RSI_RECOVERY_CONFIRMATION:
                                    if len(rsi_series_val) > rsi_cfg_recovery_lookback:
                                        past_rsi_slice = rsi_series_val.iloc[-(rsi_cfg_recovery_lookback +1) : -1]
                                        if not past_rsi_slice.empty and any(past_rsi_slice < self.config.TechnicalThresholds.RSI_OVERSOLD):
                                            rsi_recovered_from_oversold = True
                                        else:
                                            rsi_recovered_from_oversold = False
                                    else:
                                        rsi_recovered_from_oversold = False
                                else:
                                    rsi_recovered_from_oversold = False
                            else:
                                rsi_recovered_from_oversold = False
                        else:
                             logger.debug(f"Ticker '{ticker}': RSI series too short or empty for recovery detection.")
                             rsi_recovered_from_oversold = False
                    except Exception as e_rsi_rec:
                        logger.error(f"Ticker '{ticker}': Error calculating RSI series for recovery: {e_rsi_rec}", exc_info=True)
                        rsi_recovered_from_oversold = False
                else:
                    logger.debug(f"Ticker '{ticker}': Not enough data points ({len(hist['Close'])}) for RSI Recovery (needs approx {min_len_for_rsi_series}).")
                    rsi_recovered_from_oversold = False
            else:
                logger.warning(f"Ticker '{ticker}': Technical Analysis (ta) library not available. Skipping RSI Recovery calculations.")
                rsi_recovered_from_oversold = False

            if 'Volume' not in hist.columns or hist['Volume'].empty:
                logger.warning(f"Ticker '{ticker}': 'Volume' column not found in history or is empty. Skipping volume breakout calculation.")
            else:
                vol_avg_period = self.config.BreakoutTechnicalThresholds.VOLUME_AVG_PERIOD
                min_periods_vol = max(1, min(vol_avg_period // 2, vol_avg_period))

                if len(hist['Volume']) >= min_periods_vol:
                    avg_volume = hist['Volume'].rolling(window=vol_avg_period, min_periods=min_periods_vol).mean().iloc[-1]
                    current_volume_val = hist['Volume'].iloc[-1] # Renamed

                    if pd.isna(avg_volume) or avg_volume == 0:
                        logger.warning(f"Ticker '{ticker}': Average volume is NaN or zero (avg_volume: {avg_volume}). Volume ratio cannot be calculated.")
                    elif pd.isna(current_volume_val):
                        logger.warning(f"Ticker '{ticker}': Current volume is NaN. Volume ratio cannot be calculated.")
                    else:
                        volume_ratio = current_volume_val / avg_volume
                        if volume_ratio >= self.config.BreakoutTechnicalThresholds.VOLUME_SURGE_MULTIPLIER:
                            is_volume_breakout = True
                            logger.info(f"Ticker '{ticker}': Volume breakout detected! Ratio: {volume_ratio:.2f}")
                        else:
                            logger.debug(f"Ticker '{ticker}': No volume breakout. Ratio: {volume_ratio:.2f}")
                else:
                    logger.warning(f"Ticker '{ticker}': Not enough data points ({len(hist['Volume'])}) for average volume calculation (period: {vol_avg_period}, min_periods: {min_periods_vol}).")

            return_dict = {
                'ticker': ticker,
                'company_name': info.get('longName', original_ticker),
                'current_price': current_price,
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'volume': current_volume_val if 'current_volume_val' in locals() and pd.notna(current_volume_val) else (hist['Volume'].iloc[-1] if 'Volume' in hist.columns and not hist['Volume'].empty else 0),
                'rsi': rsi_val, # Use the renamed rsi_val
                'sma_20': sma_20,
                'sma_50': sma_50 if sma_50 is not None else sma_20,
                'momentum_20d': momentum_20d,
                'pe_ratio': info.get('trailingPE'),
                'roe': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'profit_margin': info.get('profitMargins'),
                'is_volume_breakout': is_volume_breakout,
                'volume_ratio': volume_ratio,
                'sma_20_50_crossed_bullish': sma_20_50_crossed_bullish,
                'sma_50_200_crossed_bullish': sma_50_200_crossed_bullish,
                'sma_20_50_crossed_bearish': sma_20_50_crossed_bearish,
                'sma_50_200_crossed_bearish': sma_50_200_crossed_bearish,
                'bb_breakout_upper': bb_breakout_upper,
                'bb_squeeze': bb_squeeze,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_ma': bb_ma,
                'sma_200': sma_200 if pd.notna(sma_200) else None,
                'breakout_n_day_high': breakout_n_day_high_signal,
                'breakdown_n_day_low': breakdown_n_day_low_signal,
                'n_day_high': n_day_high_value,
                'n_day_low': n_day_low_value,
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
        
        if not is_valid_indicator(current_price):
            logger.warning(f"Current price is None or NaN for {ticker}. Technical score might be unreliable or zero.")
        
        return max(0, min(100, score)), reasons

    def calculate_breakout_score(self, data: Dict) -> Tuple[Optional[float], List[str]]:
        score = 0.0
        reasons = []
        ticker = data.get('ticker', 'Unknown')

        if not data:
            logger.warning(f"Ticker '{ticker}': No data provided for breakout scoring.")
            return None, ["No data for breakout scoring"]

        weights = self.config.BreakoutScoringWeights

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
            n_day_high_period = self.config.TechnicalThresholds.SMA_MEDIUM
            score += weights.PRICE_BREAKOUT_S_R_WEIGHT * 100
            reasons.append(f"Price broke {n_day_high_period}-day high ({data.get('n_day_high', 0.0):.2f})")
            logger.debug(f"Ticker '{ticker}': +{weights.PRICE_BREAKOUT_S_R_WEIGHT*100} for N-day high breakout.")

        ma_crossover_scored = False
        if sma_50_200_crossed_bullish:
            score += weights.MA_CROSSOVER_WEIGHT * 100
            reasons.append("Bullish MA Crossover (SMA50 > SMA200)")
            logger.debug(f"Ticker '{ticker}': +{weights.MA_CROSSOVER_WEIGHT*100} for SMA50/200 crossover.")
            ma_crossover_scored = True
        elif sma_20_50_crossed_bullish and not ma_crossover_scored:
            score += weights.MA_CROSSOVER_WEIGHT * 0.75 * 100
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
            if bb_breakout_upper:
                bonus_squeeze_breakout = 0.05 * 100
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

        if not reasons:
            logger.info(f"Ticker '{ticker}': No breakout signals detected for breakout score.")
            return 0.0, ["No specific breakout signals detected."]

        return max(0, min(100, score)), reasons
    
    def calculate_fundamental_score(self, data: Dict) -> Tuple[float, List[str]]:
        score = 50.0
        reasons = []

        if not data:
            return 0.0, ["Input data is missing for fundamental score"] # Return float for consistency
        
        pe_ratio = data.get('pe_ratio')
        roe = data.get('roe')
        debt_to_equity = data.get('debt_to_equity')

        # Ensure fundamental metrics are valid numbers before using them
        valid_pe = isinstance(pe_ratio, (int, float)) and not (np and np.isnan(pe_ratio)) if pe_ratio is not None else False
        valid_roe = isinstance(roe, (int, float)) and not (np and np.isnan(roe)) if roe is not None else False
        valid_dte = isinstance(debt_to_equity, (int, float)) and not (np and np.isnan(debt_to_equity)) if debt_to_equity is not None else False

        if not (valid_pe and valid_roe and valid_dte):
            missing_metrics = []
            if not valid_pe: missing_metrics.append("P/E")
            if not valid_roe: missing_metrics.append("ROE")
            if not valid_dte: missing_metrics.append("D/E")
            logger.warning(f"Ticker '{data.get('ticker', 'Unknown')}': Missing or invalid fundamental metrics for scoring: {', '.join(missing_metrics)}.")
            reasons.append(f"Missing fundamental data: {', '.join(missing_metrics)}")
            return 0.0, reasons # Return 0 if critical fundamental data is missing

        if 0 < pe_ratio < self.config.FundamentalThresholds.PE_EXCELLENT_MAX:
            score += 15
            reasons.append(f"Excellent P/E ratio ({pe_ratio:.1f})")
        elif pe_ratio > self.config.FundamentalThresholds.PE_POOR_MIN:
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
        logger.info(f"Starting analysis for ticker: '{ticker}'...")
        try:
            data = self.get_stock_data(ticker)
            if not data:
                logger.warning(f"Analysis skipped for ticker '{ticker}': No data obtained or ticker invalid.")
                return None
            
            normalized_ticker = data['ticker']
            logger.info(f"Data obtained for '{normalized_ticker}'. Calculating scores...")
            technical_score, technical_reasons = self.calculate_technical_score(data)
            fundamental_score, fundamental_reasons = self.calculate_fundamental_score(data)
            breakout_score, breakout_reasons = self.calculate_breakout_score(data)
            
            hist_df_for_trading_params = data.get('hist_data_df')
            trading_parameters = self.calculate_trading_parameters(data, hist_df_for_trading_params)

            overall_score = (
                technical_score * self.config.ScoringWeights.TECHNICAL_WEIGHT +
                fundamental_score * self.config.ScoringWeights.FUNDAMENTAL_WEIGHT +
                50 * self.config.ScoringWeights.SENTIMENT_WEIGHT
            )
            
            recommendation = "HOLD" # Default
            confidence = 0.5
            if overall_score >= self.config.ScoringWeights.STRONG_BUY_THRESHOLD:
                recommendation = "STRONG_BUY"
                confidence = 0.9
            elif overall_score >= self.config.ScoringWeights.BUY_THRESHOLD:
                recommendation = "BUY"
                confidence = 0.7
            elif overall_score <= self.config.ScoringWeights.SELL_THRESHOLD: # Adjusted to make HOLD the default
                recommendation = "SELL"
                confidence = 0.7
            # This implies overall_score between SELL_THRESHOLD and HOLD_THRESHOLD (e.g. 25-40) will also be SELL
            # And scores below STRONG_SELL_THRESHOLD (implicitly anything below SELL_THRESHOLD if not defined)
            # To make HOLD the default for scores between SELL and BUY:
            # elif overall_score < self.config.ScoringWeights.BUY_THRESHOLD and overall_score > self.config.ScoringWeights.SELL_THRESHOLD:
            #    recommendation = "HOLD"
            #    confidence = 0.5
            # For now, sticking to the provided logic which prioritizes defined thresholds.
            # A specific STRONG_SELL threshold might be useful if differentiating from SELL.
            # The original logic for STRONG_SELL was:
            # else: recommendation = "STRONG_SELL"; confidence = 0.9
            # This means any score below SELL_THRESHOLD becomes STRONG_SELL. Let's assume that's intended.
            if overall_score < self.config.ScoringWeights.SELL_THRESHOLD: # Explicit STRONG_SELL
                 recommendation = "STRONG_SELL"
                 confidence = 0.9


            all_reasons = technical_reasons + fundamental_reasons
            
            # Ensure all required fields for StockAnalysis are present in `data` or have defaults
            # Many of these are now Optional in StockAnalysis, so .get() with a default or direct access is fine
            # However, for non-Optional fields in StockAnalysis that come from `data`, ensure they exist
            # For example, rsi, sma_20, sma_50, momentum_20d, pe_ratio, roe, debt_to_equity are non-Optional in dataclass
            # but are derived and could be None from get_stock_data if not calculable.
            # This should be addressed by making them Optional in StockAnalysis or ensuring default float values.
            # Forcing them to 0.0 if None for dataclass instantiation for now.

            analysis_obj = StockAnalysis(
                ticker=normalized_ticker,
                company_name=data.get('company_name', 'N/A'), # type: ignore
                current_price=data.get('current_price', 0.0), # type: ignore
                sector=data.get('sector', 'N/A'), # type: ignore
                rsi=data.get('rsi') if data.get('rsi') is not None else np.nan, # Use np.nan for float if None
                sma_20=data.get('sma_20') if data.get('sma_20') is not None else np.nan,
                sma_50=data.get('sma_50') if data.get('sma_50') is not None else np.nan,
                momentum_20d=data.get('momentum_20d') if data.get('momentum_20d') is not None else np.nan,
                pe_ratio=data.get('pe_ratio') if data.get('pe_ratio') is not None else np.nan,
                roe=data.get('roe') if data.get('roe') is not None else np.nan,
                debt_to_equity=data.get('debt_to_equity') if data.get('debt_to_equity') is not None else np.nan,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                overall_score=overall_score,
                breakout_score=breakout_score,
                recommendation=recommendation,
                confidence=confidence,
                reasons=all_reasons,
                breakout_reasons=breakout_reasons,
                is_volume_breakout=data.get('is_volume_breakout'),
                volume_ratio=data.get('volume_ratio'),
                sma_20_50_crossed_bullish=data.get('sma_20_50_crossed_bullish'),
                sma_50_200_crossed_bullish=data.get('sma_50_200_crossed_bullish'),
                sma_20_50_crossed_bearish=data.get('sma_20_50_crossed_bearish'),
                sma_50_200_crossed_bearish=data.get('sma_50_200_crossed_bearish'),
                bb_breakout_upper=data.get('bb_breakout_upper'),
                bb_squeeze=data.get('bb_squeeze'),
                bb_upper=data.get('bb_upper'),
                bb_lower=data.get('bb_lower'),
                bb_ma=data.get('bb_ma'),
                breakout_n_day_high=data.get('breakout_n_day_high'),
                breakdown_n_day_low=data.get('breakdown_n_day_low'),
                n_day_high=data.get('n_day_high'),
                n_day_low=data.get('n_day_low'),
                macd_bullish_crossover=data.get('macd_bullish_crossover'),
                macd_bearish_crossover=data.get('macd_bearish_crossover'),
                rsi_recovered_from_oversold=data.get('rsi_recovered_from_oversold'),
                macd_line=data.get('macd_line'),
                macd_signal_line=data.get('macd_signal_line'),
                suggested_entry_price=trading_parameters.get('suggested_entry_price'),
                target_price_1=trading_parameters.get('target_price_1'),
                stop_loss_price=trading_parameters.get('stop_loss_price'),
                risk_reward_ratio=trading_parameters.get('risk_reward_ratio'),
                risk_level=trading_parameters.get('risk_level'), # type: ignore
                max_position_size_pct=trading_parameters.get('max_position_size_pct'),
                trading_strategy_summary=trading_parameters.get('trading_strategy_summary'), # type: ignore
                time_horizon=trading_parameters.get('time_horizon') # type: ignore
            )
            # Fill NaN values for non-optional float fields with a default (e.g., 0.0 or specific np.nan handling)
            # This is a workaround for the dataclass non-Optional float fields if data source can provide None
            for field_name in ['rsi', 'sma_20', 'sma_50', 'momentum_20d', 'pe_ratio', 'roe', 'debt_to_equity']:
                if getattr(analysis_obj, field_name) is None or (isinstance(getattr(analysis_obj, field_name), float) and np and np.isnan(getattr(analysis_obj, field_name))):
                     # logger.warning(f"Ticker '{normalized_ticker}': Field {field_name} is None/NaN, setting to np.nan for dataclass.")
                     setattr(analysis_obj, field_name, np.nan if np else 0.0) # Use np.nan if numpy is available, else 0.0
            return analysis_obj

        except Exception as e:
            logger.error(f"Unexpected error during detailed analysis of ticker '{data.get('ticker', ticker) if data else ticker}': {e}", exc_info=True)
            return None
    
    def analyze_multiple_stocks(self, tickers: List[str]) -> List[StockAnalysis]:
        if not tickers:
            logger.warning("No tickers provided for analysis.")
            return []
        logger.info(f"Starting analysis for {len(tickers)} stock(s): {tickers}")
        results = []
        for i, ticker_input in enumerate(tickers, 1):
            logger.info(f"Processing {i}/{len(tickers)}: '{ticker_input}'")
            analysis = self.analyze_stock(ticker_input)
            if analysis:
                results.append(analysis)
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

    if not pd or not yf:
        logger.critical("Critical libraries (Pandas or yfinance) not loaded. Aborting main execution.")
        print("❌ Critical libraries (Pandas or yfinance) failed to load. Please check the log and install missing packages. Exiting.")
        return

    comprehensive_test_tickers = [
        # Large caps
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA',
        # Different sectors
        'JPM',  # Financials
        'JNJ',  # Healthcare
        'XOM',  # Energy
        'PG',   # Consumer Staples
        'KO',   # Consumer Staples
        # Various price ranges/other large caps
        'TSLA', 'META', 'V', 'WMT', 'HD',
        # Additional diverse S&P 500 stocks
        'CVX',  # Energy (another one for diversity)
        'PFE',  # Healthcare (another one)
        'BAC',  # Financials (another one)
        'CSCO', # Technology/Networking
        'PEP'   # Consumer Staples/Beverages
    ]

    analyzer = StockAnalyzer()
    logger.info(f"Starting comprehensive analysis for {len(comprehensive_test_tickers)} stocks...")
    results_list: List[StockAnalysis] = []

    for i, ticker_symbol in enumerate(comprehensive_test_tickers, 1):
        logger.info(f"Processing {i}/{len(comprehensive_test_tickers)}: {ticker_symbol}")
        analysis_result = analyzer.analyze_stock(ticker_symbol)
        if analysis_result:
            results_list.append(analysis_result)
        else:
            logger.warning(f"No analysis data returned for {ticker_symbol}. It will be excluded from the report.")

    logger.info(f"Successfully analyzed {len(results_list)} out of {len(comprehensive_test_tickers)} stocks.")

    # Process collected analysis data for summary report
    risk_level_counts = {"Conservative": 0, "Moderate": 0, "Aggressive": 0, "Undefined": 0, "Other": 0}
    tradeable_setups: List[StockAnalysis] = []
    all_risk_reward_ratios: List[float] = []
    problematic_analyses: List[str] = []

    for analysis in results_list:
        current_risk_level = analysis.risk_level if analysis.risk_level else "Undefined"
        risk_level_counts[current_risk_level] = risk_level_counts.get(current_risk_level, 0) + 1

        if current_risk_level not in ["Undefined", "Other"] and \
           analysis.suggested_entry_price is not None and \
           analysis.stop_loss_price is not None and \
           analysis.target_price_1 is not None:
            tradeable_setups.append(analysis)
            if analysis.risk_reward_ratio is not None and isinstance(analysis.risk_reward_ratio, (float, int)):
                all_risk_reward_ratios.append(analysis.risk_reward_ratio)

        if analysis.breakout_score is None:
            problematic_analyses.append(f"Ticker {analysis.ticker}: Breakout score is None despite analysis object being present.")

        if current_risk_level not in ["Undefined", "Other"] and \
           (not analysis.suggested_entry_price or not analysis.stop_loss_price or not analysis.target_price_1):
            problematic_analyses.append(f"Ticker {analysis.ticker}: Risk Level '{current_risk_level}' but missing essential trade parameters (Entry/SL/TP).")

    valid_for_top_5 = [ts for ts in tradeable_setups if ts.breakout_score is not None]
    sorted_by_breakout_score = sorted(valid_for_top_5, key=lambda x: x.breakout_score if x.breakout_score is not None else -1, reverse=True)
    top_5_breakout_candidates = sorted_by_breakout_score[:5]

    avg_rr_ratio = None
    max_rr_value = None
    min_rr_value = None
    max_rr_ticker = ""
    min_rr_ticker = ""

    if all_risk_reward_ratios:
        avg_rr_ratio = sum(all_risk_reward_ratios) / len(all_risk_reward_ratios)

    temp_max_rr = -float('inf')
    temp_min_rr = float('inf')
    for ts_setup in tradeable_setups:
        if ts_setup.risk_reward_ratio is not None:
            if ts_setup.risk_reward_ratio > temp_max_rr:
                temp_max_rr = ts_setup.risk_reward_ratio
                max_rr_ticker = ts_setup.ticker
            if ts_setup.risk_reward_ratio < temp_min_rr:
                temp_min_rr = ts_setup.risk_reward_ratio
                min_rr_ticker = ts_setup.ticker

    if temp_max_rr != -float('inf'): max_rr_value = temp_max_rr
    if temp_min_rr != float('inf'): min_rr_value = temp_min_rr

    # --- Generate Formatted Summary Report Output ---
    print("\n" + "="*60)
    print("COMPREHENSIVE BREAKOUT TRADING SYSTEM REPORT")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Stocks Scanned: {len(comprehensive_test_tickers)}")
    print(f"Successfully Processed: {len(results_list)}")
    print("="*60 + "\n")

    print("--- Risk Level Distribution ---")
    print(f"Conservative Candidates: {risk_level_counts.get('Conservative', 0)}")
    print(f"Moderate Candidates:     {risk_level_counts.get('Moderate', 0)}")
    print(f"Aggressive Candidates:   {risk_level_counts.get('Aggressive', 0)}")
    print(f"Stocks with no trade signal (Undefined Risk or low score): {risk_level_counts.get('Undefined', 0) + risk_level_counts.get('Other', 0)}")
    print("-"*(len("--- Risk Level Distribution ---")) + "\n")

    print("--- Top 5 Breakout Candidates (by Breakout Score) ---")
    if top_5_breakout_candidates:
        for i, candidate in enumerate(top_5_breakout_candidates, 1):
            print(f"{i}. Ticker: {candidate.ticker} ({candidate.company_name or 'N/A'})")
            print(f"   Breakout Score: {candidate.breakout_score:.2f}/100" if candidate.breakout_score is not None else "   Breakout Score: N/A")
            print(f"   Risk Level: {candidate.risk_level or 'N/A'}")
            print(f"   Trading Plan: {candidate.trading_strategy_summary or 'N/A'}")
            if i < len(top_5_breakout_candidates): print("   ---") # Separator
    else:
        print("No tradeable breakout candidates found with sufficient score.")
    print("-"*(len("--- Top 5 Breakout Candidates (by Breakout Score) ---")) + "\n")

    print("--- Risk/Reward Overview (for tradeable setups) ---")
    if all_risk_reward_ratios:
        print(f"Average Risk/Reward Ratio: {avg_rr_ratio:.2f}:1" if avg_rr_ratio is not None else "Average Risk/Reward Ratio: N/A")
        print(f"Highest Risk/Reward Ratio: {max_rr_value:.2f}:1 (Ticker: {max_rr_ticker})" if max_rr_value is not None and max_rr_ticker else "Highest Risk/Reward Ratio: N/A")
        print(f"Lowest Risk/Reward Ratio:  {min_rr_value:.2f}:1 (Ticker: {min_rr_ticker})" if min_rr_value is not None and min_rr_ticker else "Lowest Risk/Reward Ratio: N/A")
    else:
        print("No tradeable setups with Risk/Reward information available.")
    print("-"*(len("--- Risk/Reward Overview (for tradeable setups) ---")) + "\n")

    print("--- Issues and Observations ---")
    if problematic_analyses:
        print("Potential issues found during analysis (see logs for more details):")
        for problem in problematic_analyses:
            print(f"  - {problem}")
    else:
        print("No significant processing issues identified for successfully analyzed stocks.")
    print("-"*(len("--- Issues and Observations ---")) + "\n")

    print("="*60)
    print("End of Report")
    print("="*60)

if __name__ == "__main__":
    # Setup logger to show DEBUG messages for this test run if desired
    # logging.getLogger().setLevel(logging.DEBUG) # Example: Enable DEBUG for more verbose logs
    main()

# [end of stock_analyzer.py]
