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
    API_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
    
    # =============================================================================
    # Data Settings
    # =============================================================================
    
    # Cache settings
    CACHE_DIR = "data_cache"
    CACHE_TIMEOUT_HOURS = 6
    
    # Minimum data requirements
    MIN_DATA_POINTS = 50  # Minimum trading days for analysis
    
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
