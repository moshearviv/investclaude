#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Stock Analyzer - Core Analysis Engine
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from dataclasses import dataclass
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info("Stock Analyzer initialized")
    
    def get_sp500_tickers(self) -> List[str]:
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving S&P 500 list: {e}")
            return ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
                    'BRK-B', 'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'LLY', 'JPM', 'PG',
                    'MA', 'AVGO', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'KO', 'BAC']
          def get_stock_data(self, ticker: str) -> Optional[Dict]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="3mo")
            if hist.empty or len(hist) < self.config.MIN_DATA_POINTS:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else sma_20
            
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            momentum_20d = ((current_price / hist['Close'].iloc[-21]) - 1) * 100 if len(hist) >= 21 else 0
            
            return {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'current_price': current_price,
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'volume': hist['Volume'].iloc[-1],
                'rsi': rsi,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'momentum_20d': momentum_20d,
                'pe_ratio': info.get('trailingPE', 0),
                'roe': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'profit_margin': info.get('profitMargins', 0)
            }
        except Exception as e:
            logger.warning(f"Error getting data for {ticker}: {e}")
            return None
    
    def calculate_technical_score(self, data: Dict) -> Tuple[float, List[str]]:
        score = 50.0
        reasons = []
        
        current_price = data['current_price']
        rsi = data['rsi']
        sma_20 = data['sma_20']
        sma_50 = data['sma_50']
        momentum = data['momentum_20d']
        
        if rsi < self.config.TechnicalThresholds.RSI_OVERSOLD:
            score += 15
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.config.TechnicalThresholds.RSI_OVERBOUGHT:
            score -= 15
            reasons.append(f"RSI overbought ({rsi:.1f})")
        
        if current_price > sma_20 > sma_50:
            score += 10
            reasons.append("Price above moving averages")
        elif current_price < sma_20 < sma_50:
            score -= 10
            reasons.append("Price below moving averages")
        
        if momentum > 5:
            score += 10
            reasons.append(f"Strong positive momentum ({momentum:.1f}%)")
        elif momentum < -5:
            score -= 10
            reasons.append(f"Negative momentum ({momentum:.1f}%)")
        
        return max(0, min(100, score)), reasons
    
    def calculate_fundamental_score(self, data: Dict) -> Tuple[float, List[str]]:
        score = 50.0
        reasons = []
        
        pe_ratio = data['pe_ratio']
        roe = data['roe']
        debt_to_equity = data['debt_to_equity']
        
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
        try:
            logger.info(f"Analyzing {ticker}...")
            data = self.get_stock_data(ticker)
            if not data:
                return None
            
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
                ticker=ticker,
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
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    def analyze_multiple_stocks(self, tickers: List[str]) -> List[StockAnalysis]:
        logger.info(f"Starting analysis of {len(tickers)} stocks...")
        results = []
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {i}/{len(tickers)}: {ticker}")
            analysis = self.analyze_stock(ticker)
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
    test_tickers = ['AAPL', 'MSFT', 'GOOG']
    print(f"Testing analyzer with {test_tickers}")
    
    results = analyzer.analyze_multiple_stocks(test_tickers)
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"Analyzed {len(results)} stocks")
        for result in results:
            print(f"\n{result.ticker} ({result.company_name}):")
            print(f"  Score: {result.overall_score:.1f}/100")
            print(f"  Recommendation: {result.recommendation}")
            print(f"  Price: ${result.current_price:.2f}")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()
