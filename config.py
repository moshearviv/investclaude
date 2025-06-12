#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Configuration for Advanced Stock Analysis System
הגדרות בסיסיות למערכת ניתוח מניות מתקדמת
"""

import os
from datetime import datetime

class Config:
    """Basic configuration settings"""
    
    # =============================================================================
    # Basic Settings
    # =============================================================================
    
    # Default analysis parameters
    DEFAULT_NUM_STOCKS = 50
    MIN_STOCKS = 10
    MAX_STOCKS = 500
    
    DEFAULT_MIN_SCORE = 60
    DEFAULT_MIN_CONFIDENCE = 0.6
    
    # Performance settings
    DEFAULT_MAX_WORKERS = 5
    API_TIMEOUT = 30 # seconds
    RETRY_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 1 # seconds between retries for yfinance calls
    
    # =============================================================================
    # Data Settings
    # =============================================================================
    
    # Cache settings
    CACHE_DIR = "data_cache"
    CACHE_TIMEOUT_HOURS = 6
    
    # Minimum data requirements
    MIN_DATA_POINTS = 50  # Minimum trading days for analysis

    # Fallback ticker list if S&P 500 retrieval fails
    FALLBACK_TICKERS = [
        'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
        'BRK-B', 'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'LLY', 'JPM', 'PG',
        'MA', 'AVGO', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'KO', 'BAC'
    ]
    
    # =============================================================================
    # API Settings (Optional)
    # =============================================================================
    
    # Twitter API (optional for sentiment analysis)
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    
    # =============================================================================
    # Analysis Thresholds
    # =============================================================================
    
    class TechnicalThresholds:
        """Technical analysis thresholds"""
        RSI_OVERSOLD = 30
        RSI_OVERBOUGHT = 70
        
        SMA_SHORT = 20
        SMA_MEDIUM = 50
        SMA_LONG = 200

    class BreakoutTechnicalThresholds:
        """Parameters for breakout detection logic"""
        # Volume Breakout
        VOLUME_AVG_PERIOD: int = 50         # Average volume calculation period
        VOLUME_SURGE_MULTIPLIER: float = 2.0  # Multiplier for detecting volume spike

        # Bollinger Bands
        BB_WINDOW: int = 20                 # Window for BB calculation
        BB_STD_DEV: int = 2                 # Number of standard deviations for BB
        # Simpler BB Squeeze: if BB width is less than X times the price's stddev over BB_WINDOW
        BB_SQUEEZE_STD_THRESHOLD: float = 0.5 # Lower values indicate tighter squeeze relative to price volatility

        # MACD
        MACD_FAST_PERIOD: int = 12
        MACD_SLOW_PERIOD: int = 26
        MACD_SIGNAL_PERIOD: int = 9

        # RSI Recovery (use existing RSI_OVERSOLD from TechnicalThresholds)
        RSI_RECOVERY_CONFIRMATION: int = 35 # RSI must cross above this after being oversold
    
    class FundamentalThresholds:
        """Fundamental analysis thresholds"""
        PE_EXCELLENT_MAX = 15
        PE_GOOD_MAX = 25
        PE_POOR_MIN = 35
        
        ROE_EXCELLENT_MIN = 0.15
        ROE_GOOD_MIN = 0.10
    
    class ScoringWeights:
        """Scoring system weights"""
        TECHNICAL_WEIGHT = 0.4
        FUNDAMENTAL_WEIGHT = 0.4
        SENTIMENT_WEIGHT = 0.2
        
        # Recommendation thresholds
        STRONG_BUY_THRESHOLD = 75
        BUY_THRESHOLD = 60
        HOLD_THRESHOLD = 40
        SELL_THRESHOLD = 25

    class BreakoutScoringWeights:
        """Weights for individual breakout signals - how much each signal contributes to a 'breakout score'"""
        # These are relative strengths of signals, not necessarily summing to 1 unless part of a combined score.
        VOLUME_BREAKOUT_WEIGHT: float = 0.15
        PRICE_BREAKOUT_S_R_WEIGHT: float = 0.20  # Placeholder for Support/Resistance breakout
        MA_CROSSOVER_WEIGHT: float = 0.20        # e.g., Price crossing above SMA50 or SMA20/SMA50 crossover
        BB_BREAKOUT_WEIGHT: float = 0.15         # Price breaking above upper Bollinger Band
        BB_SQUEEZE_WEIGHT: float = 0.10          # Strength of the Bollinger Band Squeeze itself (potential energy)
        MACD_CROSSOVER_WEIGHT: float = 0.15      # MACD line crossing above signal line
        RSI_RECOVERY_WEIGHT: float = 0.05        # RSI recovering from oversold condition

    # =============================================================================
    # Trading Settings
    # =============================================================================
    class TradingSettings:
        """Parameters for defining trade execution and risk management strategies."""

        # Risk Level Breakout Score Thresholds: Minimum breakout_score needed to consider a trade
        CONSERVATIVE_BREAKOUT_SCORE_MIN: int = 80
        MODERATE_BREAKOUT_SCORE_MIN: int = 60
        AGGRESSIVE_BREAKOUT_SCORE_MIN: int = 40

        # Stop-Loss Percentages (from entry price)
        CONSERVATIVE_STOP_LOSS_PCT: float = 0.03  # 3% stop-loss
        MODERATE_STOP_LOSS_PCT: float = 0.05    # 5% stop-loss
        AGGRESSIVE_STOP_LOSS_PCT: float = 0.08    # 8% stop-loss

        # Target Profit Percentages (from entry price)
        CONSERVATIVE_TARGET_PROFIT_PCT: float = 0.06  # 6% target profit (2:1 R/R with 3% stop)
        MODERATE_TARGET_PROFIT_PCT: float = 0.10    # 10% target profit (2:1 R/R with 5% stop)
        AGGRESSIVE_TARGET_PROFIT_PCT: float = 0.15    # 15% target profit (~2:1 R/R with 8% stop)
        # Note: These are initial target percentages. Risk/Reward will also be calculated.

        # Max Position Size Percentages (of hypothetical total portfolio value)
        CONSERVATIVE_MAX_POS_SIZE_PCT: float = 0.02  # Max 2% of portfolio in one trade
        MODERATE_MAX_POS_SIZE_PCT: float = 0.04    # Max 4% of portfolio in one trade
        AGGRESSIVE_MAX_POS_SIZE_PCT: float = 0.06    # Max 6% of portfolio in one trade

        # ATR Configuration (for dynamic stop-loss/take-profit, future use)
        ATR_PERIOD: int = 14                             # Period for ATR calculation
        ATR_STOP_MULTIPLIER: float = 2.0                 # e.g., Entry Price - (2 * ATR)
        ATR_TARGET_MULTIPLIER_RR: float = 3.0            # e.g., If ATR stop is X, target profit is X * Multiplier (for 3:1 R/R)

        # Default Time Horizon for trades suggested by this system
        DEFAULT_TRADE_TIME_HORIZON: str = "Short-term (1-4 weeks)"


# Helper functions
def create_cache_dir():
    """Create cache directory if it doesn't exist"""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

def get_config_summary():
    """Return configuration summary"""
    return {
        'default_stocks': Config.DEFAULT_NUM_STOCKS,
        'min_score': Config.DEFAULT_MIN_SCORE,
        'cache_hours': Config.CACHE_TIMEOUT_HOURS,
        'twitter_enabled': bool(Config.TWITTER_BEARER_TOKEN)
    }

# Create cache directory on import
create_cache_dir()

if __name__ == "__main__":
    print("📊 Stock Analysis System Configuration")
    print("=" * 40)
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
