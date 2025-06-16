#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fundamental Analyzer for Stock Analysis System
"""

import pandas as pd
import yfinance as yf # Ensure yfinance is imported
import numpy as np # Added for potential NaN checks and calculations
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    def __init__(self):
        """
        Initializes the FundamentalAnalyzer.
        """
        pass

    def get_financial_data(self, yf_ticker_obj: yf.Ticker) -> dict:
        """
        Fetches quarterly financials, earnings, cash flow, and balance sheet.
        """
        logger.debug(f"Attempting to fetch financial statements via yf_ticker_obj...")
        financials_data = {
            'quarterly_financials': None,
            'quarterly_earnings': None,
            'quarterly_cashflow': None,
            'quarterly_balance_sheet': None
        }
        try:
            financials_data['quarterly_financials'] = yf_ticker_obj.quarterly_financials
            financials_data['quarterly_earnings'] = yf_ticker_obj.quarterly_earnings
            financials_data['quarterly_cashflow'] = yf_ticker_obj.quarterly_cashflow
            financials_data['quarterly_balance_sheet'] = yf_ticker_obj.quarterly_balance_sheet
            logger.debug(f"Successfully fetched financial statements for ticker via yf_ticker_obj.")
        except Exception as e:
            logger.error(f"Error fetching financial statements via yf_ticker_obj: {e}")
            # Ensure all keys still exist even if fetching fails, with None values
            financials_data = {key: None for key in financials_data}
        return financials_data

    def calculate_growth_rates(self, financials_df: pd.DataFrame, earnings_df: pd.DataFrame) -> dict:
        """
        Calculates QoQ and YoY growth rates for revenue and earnings.
        Assumes DataFrames have time periods as columns, sorted descending by date.
        """
        growth_data = {
            'revenue_growth_yoy': None,
            'revenue_growth_qoq': None,
            'earnings_growth_yoy': None,
            'earnings_growth_qoq': None,
        }

        try:
            # Revenue Growth
            if financials_df is not None and not financials_df.empty and 'Total Revenue' in financials_df.index:
                revenue_series = financials_df.loc['Total Revenue'].dropna()
                if len(revenue_series) >= 2: # Need at least two points for QoQ
                    current_revenue = revenue_series.iloc[0]
                    prev_q_revenue = revenue_series.iloc[1]
                    if pd.notna(current_revenue) and pd.notna(prev_q_revenue) and prev_q_revenue != 0:
                        growth_data['revenue_growth_qoq'] = (current_revenue - prev_q_revenue) / abs(prev_q_revenue)
                    else:
                        logger.warning("Could not calculate QoQ revenue growth due to missing or zero values.")
                else:
                    logger.warning("Insufficient data for QoQ revenue growth (need at least 2 quarters).")

                if len(revenue_series) >= 5: # Need at least five points for YoY (current vs. 4 quarters ago)
                    current_revenue = revenue_series.iloc[0]
                    year_ago_revenue = revenue_series.iloc[4]
                    if pd.notna(current_revenue) and pd.notna(year_ago_revenue) and year_ago_revenue != 0:
                        growth_data['revenue_growth_yoy'] = (current_revenue - year_ago_revenue) / abs(year_ago_revenue)
                    else:
                        logger.warning("Could not calculate YoY revenue growth due to missing or zero values.")
                else:
                    logger.warning("Insufficient data for YoY revenue growth (need at least 5 quarters).")
            else:
                logger.warning("Financials data is missing or 'Total Revenue' not found for growth rate calculation.")

            # Earnings Growth
            if earnings_df is not None and not earnings_df.empty and 'Earnings' in earnings_df.index:
                earnings_series = earnings_df.loc['Earnings'].dropna()
                if len(earnings_series) >= 2:
                    current_earnings = earnings_series.iloc[0]
                    prev_q_earnings = earnings_series.iloc[1]
                    if pd.notna(current_earnings) and pd.notna(prev_q_earnings) and prev_q_earnings != 0:
                        growth_data['earnings_growth_qoq'] = (current_earnings - prev_q_earnings) / abs(prev_q_earnings)
                    else:
                        logger.warning("Could not calculate QoQ earnings growth due to missing or zero values.")

                else:
                    logger.warning("Insufficient data for QoQ earnings growth (need at least 2 quarters).")

                if len(earnings_series) >= 5:
                    current_earnings = earnings_series.iloc[0]
                    year_ago_earnings = earnings_series.iloc[4]
                    if pd.notna(current_earnings) and pd.notna(year_ago_earnings) and year_ago_earnings != 0:
                        growth_data['earnings_growth_yoy'] = (current_earnings - year_ago_earnings) / abs(year_ago_earnings)
                    else:
                        logger.warning("Could not calculate YoY earnings growth due to missing or zero values.")
                else:
                    logger.warning("Insufficient data for YoY earnings growth (need at least 5 quarters).")
            else:
                logger.warning("Earnings data is missing or 'Earnings' not found for growth rate calculation.")
        except KeyError as e:
            logger.error(f"KeyError during growth rate calculation: {e}. Check DataFrame structure.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during growth rate calculation: {e}")

        return growth_data

    def analyze_free_cash_flow_trends(self, cashflow_df: pd.DataFrame) -> dict:
        """
        Analyzes free cash flow trends.
        Assumes cashflow_df has time periods as columns, sorted descending by date.
        """
        fcf_data = {
            'free_cash_flow_latest': None,
            'free_cash_flow_trend': None,
            'free_cash_flow_consistently_positive': None
        }

        try:
            if cashflow_df is not None and not cashflow_df.empty and 'Free Cash Flow' in cashflow_df.index:
                fcf_series = cashflow_df.loc['Free Cash Flow'].dropna().astype(float)

                if not fcf_series.empty:
                    fcf_data['free_cash_flow_latest'] = fcf_series.iloc[0]

                    if len(fcf_series) >= 2:
                        if fcf_series.iloc[0] > fcf_series.iloc[1]:
                            fcf_data['free_cash_flow_trend'] = 'positive'
                        elif fcf_series.iloc[0] < fcf_series.iloc[1]:
                            fcf_data['free_cash_flow_trend'] = 'negative'
                        else:
                            fcf_data['free_cash_flow_trend'] = 'flat'
                    elif len(fcf_series) == 1:
                         fcf_data['free_cash_flow_trend'] = 'N/A (single data point)'
                    else:
                        logger.warning("Insufficient data for FCF trend analysis (need at least 2 quarters).")


                    if len(fcf_series) >= 4:
                        fcf_data['free_cash_flow_consistently_positive'] = all(fcf_series.iloc[i] > 0 for i in range(4))
                    else:
                        logger.warning("Insufficient data for FCF consistency check (need at least 4 quarters).")
                else:
                    logger.warning("'Free Cash Flow' series is empty after dropping NaNs.")
            else:
                logger.warning("Cashflow data is missing or 'Free Cash Flow' not found for trend analysis.")
        except KeyError as e:
            logger.error(f"KeyError during FCF analysis: {e}. Check DataFrame structure.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during FCF analysis: {e}")

        return fcf_data

    def analyze_profit_margin_trends(self, financials_df: pd.DataFrame) -> dict:
        """
        Analyzes profit margin trends.
        Assumes financials_df has time periods as columns, sorted descending by date.
        """
        margin_data = {
            'net_profit_margin_latest': None,
            'profit_margin_trend': None,
        }

        try:
            if financials_df is not None and not financials_df.empty and \
               'Net Income' in financials_df.index and 'Total Revenue' in financials_df.index:

                net_income_series = financials_df.loc['Net Income'].dropna().astype(float)
                total_revenue_series = financials_df.loc['Total Revenue'].dropna().astype(float)

                # Align series by index (dates) and calculate margin
                aligned_income, aligned_revenue = net_income_series.align(total_revenue_series, join='inner')

                if not aligned_revenue.empty and not aligned_income.empty:
                    # Avoid division by zero or with NaN
                    npm_series = aligned_income.divide(aligned_revenue.replace(0, np.nan)).dropna()

                    if not npm_series.empty:
                        margin_data['net_profit_margin_latest'] = npm_series.iloc[0]

                        if len(npm_series) >= 2:
                            if npm_series.iloc[0] > npm_series.iloc[1]:
                                margin_data['profit_margin_trend'] = 'positive'
                            elif npm_series.iloc[0] < npm_series.iloc[1]:
                                margin_data['profit_margin_trend'] = 'negative'
                            else:
                                margin_data['profit_margin_trend'] = 'flat'
                        elif len(npm_series) == 1:
                            margin_data['profit_margin_trend'] = 'N/A (single data point)'
                        else:
                           logger.warning("Insufficient data for profit margin trend analysis (need at least 2 quarters of calculated margins).")
                    else:
                        logger.warning("Net profit margin series is empty after calculations.")
                else:
                    logger.warning("Could not align Net Income and Total Revenue series for profit margin calculation.")
            else:
                logger.warning("Financials data is missing or 'Net Income'/'Total Revenue' not found for margin analysis.")
        except KeyError as e:
            logger.error(f"KeyError during profit margin analysis: {e}. Check DataFrame structure.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during profit margin analysis: {e}")

        return margin_data

    def analyze(self, ticker: str, yf_ticker_obj: yf.Ticker) -> dict:
        """
        Main analysis method for fundamental data.
        Returns a dictionary of all calculated fundamental metrics.
        """
        logger.info(f"Starting fundamental analysis for {ticker}...")
        all_fundamental_data = {}

        raw_financial_data = self.get_financial_data(yf_ticker_obj)

        q_financials = raw_financial_data.get('quarterly_financials')
        q_earnings = raw_financial_data.get('quarterly_earnings')
        q_cashflow = raw_financial_data.get('quarterly_cashflow')
        # q_balance_sheet = raw_financial_data.get('quarterly_balance_sheet') # Available if needed

        # Growth Rates Calculation
        if q_financials is not None and not q_financials.empty and \
           q_earnings is not None and not q_earnings.empty:
            growth_rates = self.calculate_growth_rates(q_financials, q_earnings)
            all_fundamental_data.update(growth_rates)
        else:
            logger.warning(f"Ticker {ticker}: Skipping growth rate calculation due to missing quarterly financials or earnings data.")
            # Still add keys with None to maintain structure if desired, or rely on method's default None
            all_fundamental_data.update({
                'revenue_growth_yoy': None, 'revenue_growth_qoq': None,
                'earnings_growth_yoy': None, 'earnings_growth_qoq': None
            })


        # FCF Trends Analysis
        if q_cashflow is not None and not q_cashflow.empty:
            fcf_trends = self.analyze_free_cash_flow_trends(q_cashflow)
            all_fundamental_data.update(fcf_trends)
        else:
            logger.warning(f"Ticker {ticker}: Skipping FCF trend analysis due to missing quarterly cashflow data.")
            all_fundamental_data.update({
                'free_cash_flow_latest': None, 'free_cash_flow_trend': None,
                'free_cash_flow_consistently_positive': None
            })

        # Profit Margin Trends Analysis
        if q_financials is not None and not q_financials.empty:
            margin_trends = self.analyze_profit_margin_trends(q_financials)
            all_fundamental_data.update(margin_trends)
        else:
            logger.warning(f"Ticker {ticker}: Skipping profit margin trend analysis due to missing quarterly financials data.")
            all_fundamental_data.update({
                'net_profit_margin_latest': None, 'profit_margin_trend': None
            })

        logger.info(f"Fundamental analysis for {ticker} complete. Found {len(all_fundamental_data)} data points.")
        return all_fundamental_data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    analyzer = FundamentalAnalyzer()
    sample_ticker_symbol = 'MSFT'
    logger.info(f"Testing FundamentalAnalyzer for {sample_ticker_symbol} (structure only)...")
    # In later steps, we'll pass a real yf.Ticker object
    # For now, this test will be very basic as methods are placeholders.
    # results = analyzer.analyze(sample_ticker_symbol, None) # Passing None for yf_ticker_obj for now
    # logger.info(f"Results for {sample_ticker_symbol}: {results}")
    logger.info("FundamentalAnalyzer basic structure test complete.")
