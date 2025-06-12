#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Advanced Stock Breakout Analysis System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import logging
import time

# Import your analyzer
from stock_analyzer import StockAnalyzer, StockAnalysis
from config import Config

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Breakout Hunter",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .breakout-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .analysis-summary {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .stock-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .score-excellent { color: #28a745; font-weight: bold; font-size: 1.2em; }
    .score-good { color: #17a2b8; font-weight: bold; font-size: 1.1em; }
    .score-average { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def get_score_color_class(score):
    """Get CSS class based on score"""
    if score >= 75:
        return "score-excellent"
    elif score >= 60:
        return "score-good"
    elif score >= 40:
        return "score-average"
    else:
        return "score-poor"

def create_breakout_score_chart(analyses):
    """Create breakout score distribution chart"""
    if not analyses:
        return None
    
    scores = [a.breakout_score for a in analyses if a.breakout_score is not None]
    tickers = [a.ticker for a in analyses if a.breakout_score is not None]
    
    if not scores:
        return None
    
    fig = px.bar(
        x=tickers,
        y=scores,
        title="📊 Breakout Scores Across Analyzed Stocks",
        labels={'x': 'Stock Ticker', 'y': 'Breakout Score'},
        color=scores,
        color_continuous_scale='RdYlGn'
    )
    
    # Add threshold lines
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Conservative Threshold")
    fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Moderate Threshold")
    fig.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Aggressive Threshold")
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

def display_stock_analysis(analysis: StockAnalysis, rank: int):
    """Display detailed analysis for a single stock"""
    
    # Determine card style based on breakout score
    breakout_score = analysis.breakout_score if analysis.breakout_score else 0
    
    if breakout_score >= 60:
        card_class = "breakout-card"
        emoji = "🚀"
    elif breakout_score >= 40:
        card_class = "warning-card"
        emoji = "⚠️"
    else:
        card_class = "stock-card"
        emoji = "📊"
    
    st.markdown(f"""
    <div class="{card_class}">
        <h3>{emoji} #{rank} - {analysis.ticker} - {analysis.company_name}</h3>
        <p><strong>Current Price:</strong> ${analysis.current_price:.2f} | <strong>Sector:</strong> {analysis.sector}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Breakout Score", 
            f"{breakout_score:.1f}/100" if breakout_score else "N/A",
            help="Higher scores indicate stronger breakout signals"
        )
        
    with col2:
        st.metric(
            "Risk Level", 
            analysis.risk_level or "N/A",
            help="Conservative/Moderate/Aggressive based on breakout strength"
        )
    
    with col3:
        if analysis.target_price_1:
            upside = ((analysis.target_price_1 / analysis.current_price) - 1) * 100
            st.metric(
                "Upside Potential", 
                f"+{upside:.1f}%",
                help=f"Target: ${analysis.target_price_1:.2f}"
            )
        else:
            st.metric("Upside Potential", "N/A")
    
    with col4:
        if analysis.risk_reward_ratio:
            st.metric(
                "Risk/Reward", 
                f"{analysis.risk_reward_ratio:.1f}:1",
                help="Higher ratios indicate better risk-adjusted returns"
            )
        else:
            st.metric("Risk/Reward", "N/A")
    
    # Breakout signals detected
    if analysis.breakout_reasons:
        st.write("**🎯 Breakout Signals Detected:**")
        for reason in analysis.breakout_reasons:
            st.write(f"• {reason}")
    
    # Trading strategy
    if analysis.trading_strategy_summary:
        st.write("**📋 Trading Strategy:**")
        st.write(analysis.trading_strategy_summary)
    
    # Technical details in expander
    with st.expander(f"📈 Technical Details for {analysis.ticker}"):
        tcol1, tcol2 = st.columns(2)
        
        with tcol1:
            st.write("**Technical Indicators:**")
            st.write(f"RSI: {analysis.rsi:.1f}" if not np.isnan(analysis.rsi) else "RSI: N/A")
            st.write(f"SMA 20: ${analysis.sma_20:.2f}" if not np.isnan(analysis.sma_20) else "SMA 20: N/A")
            st.write(f"SMA 50: ${analysis.sma_50:.2f}" if not np.isnan(analysis.sma_50) else "SMA 50: N/A")
            st.write(f"20-day Momentum: {analysis.momentum_20d:.1f}%" if not np.isnan(analysis.momentum_20d) else "Momentum: N/A")
        
        with tcol2:
            st.write("**Fundamental Metrics:**")
            st.write(f"P/E Ratio: {analysis.pe_ratio:.1f}" if not np.isnan(analysis.pe_ratio) else "P/E: N/A")
            st.write(f"ROE: {analysis.roe:.1%}" if not np.isnan(analysis.roe) else "ROE: N/A")
            st.write(f"Debt/Equity: {analysis.debt_to_equity:.2f}" if not np.isnan(analysis.debt_to_equity) else "D/E: N/A")
    
    st.markdown("---")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Stock Breakout Hunter</h1>', unsafe_allow_html=True)
    st.markdown("### Find S&P 500 stocks ready for breakout with technical, fundamental & sentiment analysis")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Number of stocks to analyze
    num_stocks = st.sidebar.slider(
        "Number of S&P 500 stocks to analyze",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="More stocks = longer analysis time but more opportunities"
    )
    
    # Minimum breakout score filter
    min_breakout_score = st.sidebar.slider(
        "Minimum breakout score to display",
        min_value=0,
        max_value=100,
        value=30,
        step=10,
        help="Only show stocks with breakout scores above this threshold"
    )
    
    # Risk level filter
    risk_levels = st.sidebar.multiselect(
        "Show risk levels",
        ["Conservative", "Moderate", "Aggressive"],
        default=["Conservative", "Moderate", "Aggressive"],
        help="Filter results by risk tolerance"
    )
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'last_analysis_time' not in st.session_state:
        st.session_state.last_analysis_time = None
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Hunt for Breakouts!", type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🚀 Initializing breakout hunter...")
            
            try:
                # Initialize analyzer
                analyzer = StockAnalyzer()
                progress_bar.progress(10)
                
                # Get S&P 500 tickers
                status_text.text("📋 Getting S&P 500 stock list...")
                all_tickers = analyzer.get_sp500_tickers()
                tickers_to_analyze = all_tickers[:num_stocks]
                progress_bar.progress(20)
                
                # Run analysis
                status_text.text(f"🔍 Analyzing {len(tickers_to_analyze)} stocks for breakout patterns...")
                
                start_time = time.time()
                results = analyzer.analyze_multiple_stocks(tickers_to_analyze)
                end_time = time.time()
                
                progress_bar.progress(100)
                
                # Store results
                st.session_state.analysis_results = results
                st.session_state.last_analysis_time = datetime.now()
                
                status_text.text("✅ Analysis complete!")
                time.sleep(1)  # Show success message briefly
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"🎉 Analyzed {len(results)} stocks in {end_time - start_time:.1f} seconds!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return
    
    # Display results if available
    if st.session_state.analysis_results:
        
        results = st.session_state.analysis_results
        
        # Filter results
        filtered_results = []
        for analysis in results:
            breakout_score = analysis.breakout_score if analysis.breakout_score else 0
            risk_level = analysis.risk_level or "Unknown"
            
            if breakout_score >= min_breakout_score and (not risk_levels or risk_level in risk_levels or risk_level == "Unknown"):
                filtered_results.append(analysis)
        
        # Sort by breakout score
        filtered_results.sort(key=lambda x: x.breakout_score if x.breakout_score else 0, reverse=True)
        
        # Summary metrics
        st.markdown("---")
        st.subheader("📊 Analysis Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Analyzed", len(results))
        
        with col2:
            breakout_candidates = len([r for r in results if r.breakout_score and r.breakout_score >= 40])
            st.metric("Breakout Candidates", breakout_candidates)
        
        with col3:
            high_confidence = len([r for r in results if r.breakout_score and r.breakout_score >= 70])
            st.metric("High Confidence", high_confidence)
        
        with col4:
            avg_score = np.mean([r.breakout_score for r in results if r.breakout_score])
            st.metric("Avg Breakout Score", f"{avg_score:.1f}" if not np.isnan(avg_score) else "N/A")
        
        with col5:
            tradeable = len([r for r in filtered_results if r.risk_level and r.risk_level != "Undefined"])
            st.metric("Tradeable Setups", tradeable)
        
        # Breakout score chart
        if results:
            fig = create_breakout_score_chart(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Main results tabs
        tab1, tab2, tab3 = st.tabs(["🚀 Top Breakout Candidates", "📊 All Results", "📈 Market Overview"])
        
        with tab1:
            st.header("🚀 Top Breakout Candidates")
            
            if filtered_results:
                st.write(f"Showing top {len(filtered_results)} candidates meeting your criteria:")
                
                # Display top candidates
                for i, analysis in enumerate(filtered_results[:10], 1):  # Top 10
                    display_stock_analysis(analysis, i)
                    
            else:
                st.info("No stocks meet your current filtering criteria. Try lowering the minimum breakout score or adjusting risk level filters.")
        
        with tab2:
            st.header("📊 All Analysis Results")
            
            if results:
                # Create summary dataframe
                summary_data = []
                for analysis in results:
                    summary_data.append({
                        'Ticker': analysis.ticker,
                        'Company': analysis.company_name[:30] + "..." if len(analysis.company_name) > 30 else analysis.company_name,
                        'Current Price': f"${analysis.current_price:.2f}",
                        'Breakout Score': f"{analysis.breakout_score:.1f}" if analysis.breakout_score else "N/A",
                        'Risk Level': analysis.risk_level or "N/A",
                        'Sector': analysis.sector,
                        'Target Price': f"${analysis.target_price_1:.2f}" if analysis.target_price_1 else "N/A",
                        'Upside %': f"+{((analysis.target_price_1/analysis.current_price-1)*100):.1f}%" if analysis.target_price_1 else "N/A",
                        'Risk/Reward': f"{analysis.risk_reward_ratio:.1f}:1" if analysis.risk_reward_ratio else "N/A"
                    })
                
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f'breakout_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                    mime='text/csv'
                )
        
        with tab3:
            st.header("📈 Market Overview")
            
            # Sector analysis
            if results:
                sectors = {}
                sector_scores = {}
                
                for analysis in results:
                    sector = analysis.sector
                    score = analysis.breakout_score if analysis.breakout_score else 0
                    
                    if sector in sectors:
                        sectors[sector] += 1
                        sector_scores[sector].append(score)
                    else:
                        sectors[sector] = 1
                        sector_scores[sector] = [score]
                
                # Sector distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_sectors = px.pie(
                        values=list(sectors.values()),
                        names=list(sectors.keys()),
                        title="Sector Distribution"
                    )
                    st.plotly_chart(fig_sectors, use_container_width=True)
                
                with col2:
                    # Average breakout scores by sector
                    avg_sector_scores = {sector: np.mean(scores) for sector, scores in sector_scores.items()}
                    
                    fig_sector_scores = px.bar(
                        x=list(avg_sector_scores.keys()),
                        y=list(avg_sector_scores.values()),
                        title="Average Breakout Score by Sector",
                        labels={'x': 'Sector', 'y': 'Avg Breakout Score'}
                    )
                    st.plotly_chart(fig_sector_scores, use_container_width=True)
        
        # Last analysis info
        if st.session_state.last_analysis_time:
            st.markdown("---")
            st.caption(f"Last analysis: {st.session_state.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        # Welcome screen
        st.markdown("---")
        st.subheader("🎯 Welcome to the Breakout Hunter!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="analysis-summary">
            <h4>📈 Technical Analysis</h4>
            <ul>
                <li>Volume breakout detection</li>
                <li>Moving average crossovers</li>
                <li>Bollinger Band squeezes</li>
                <li>MACD momentum signals</li>
                <li>RSI recovery patterns</li>
                <li>Support/resistance breaks</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-summary">
            <h4>📊 Risk Management</h4>
            <ul>
                <li>Conservative risk level</li>
                <li>Moderate risk level</li>
                <li>Aggressive risk level</li>
                <li>Stop loss recommendations</li>
                <li>Target price calculations</li>
                <li>Risk/reward ratios</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="analysis-summary">
            <h4>🎯 Trading Ready</h4>
            <ul>
                <li>Entry price suggestions</li>
                <li>Position sizing guidance</li>
                <li>Time horizon recommendations</li>
                <li>Complete trading strategy</li>
                <li>CSV export for tracking</li>
                <li>Real-time S&P 500 scanning</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("👆 Configure your settings in the sidebar and click 'Hunt for Breakouts!' to start finding trading opportunities!")

if __name__ == "__main__":
    main()
