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
from typing import Dict, List, Optional, Tuple
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

class StockAnalyzer:
    def __init__(self):
        self.config = Config()
        if pd and np and yf and requests and ta: # Check if all were imported
            logger.info("All major libraries loaded successfully.")
        else:
            logger.warning("One or more critical libraries failed to load. Functionality may be limited.")
        logger.info("Stock Analyzer initialized")
    
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

            return {
                'ticker': ticker, # Normalized ticker
                'company_name': info.get('longName', original_ticker), # Use original ticker for company name if longName is missing
                'current_price': current_price,
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns and not hist['Volume'].empty else 0,
                'rsi': rsi,
                'sma_20': sma_20,
                'sma_50': sma_50 if sma_50 is not None else sma_20, # Fallback sma_50 to sma_20 if it was not calculable
                'momentum_20d': momentum_20d,
                'pe_ratio': info.get('trailingPE'), # Allow None for PE
                'roe': info.get('returnOnEquity'), # Allow None
                'debt_to_equity': info.get('debtToEquity'), # Allow None
                'profit_margin': info.get('profitMargins') # Allow None
            }
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
            
            overall_score = (
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
                recommendation=recommendation,
                confidence=confidence,
                reasons=all_reasons
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

    # Test tickers including a bad one and a less common one like BRK-A
    test_tickers = ['AAPL', 'MSFT', 'GOOG', 'BADTICKER', 'TSLA', 'NVDA', 'BRK-A']

    print(f"🧪 Starting test analysis for tickers: {test_tickers}")
    
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
            print(f"  RSI: {found_analysis.rsi:.2f}" if found_analysis.rsi is not None else "  RSI: N/A")
            print(f"  SMA20: {found_analysis.sma_20:.2f}" if found_analysis.sma_20 is not None else "  SMA20: N/A")
            print(f"  SMA50: {found_analysis.sma_50:.2f}" if found_analysis.sma_50 is not None else "  SMA50: N/A")
            print(f"  Momentum(20d): {found_analysis.momentum_20d:.2f}%" if found_analysis.momentum_20d is not None else "  Momentum(20d): N/A")
            print(f"  P/E: {found_analysis.pe_ratio:.2f}" if found_analysis.pe_ratio is not None else "  P/E: N/A")
            print(f"  Overall Score: {found_analysis.overall_score:.1f}/100")
            print(f"  Recommendation: {found_analysis.recommendation} (Confidence: {found_analysis.confidence:.0%})")
            if len(found_analysis.reasons) > 0:
                print(f"  Reasons: {'; '.join(found_analysis.reasons)}")
        else:
            # This ticker_input was not successfully analyzed. Logs should indicate why.
            print(f"\n❌ {ticker_input.upper()}: Analysis failed or skipped (see logs for details).")

    print("-" * 30)
    print(f"📈 Test analysis completed. Successfully analyzed {len(analyzed_results)} out of {len(test_tickers)} tickers.")

if __name__ == "__main__":
    # Setup logger to show DEBUG messages for this test run if desired
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
