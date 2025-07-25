import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from data_fetcher import DataFetcher
from sentiment_analyzer import SentimentAnalyzer
from predictor import StockPredictor
from visualizer import Visualizer

# Configure Streamlit page
st.set_page_config(
    page_title="Stock/Forex Predictor",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 S.M.A.R.T. Sentiment Market Analysis and Response Toolkit")
    st.markdown("### Predict market movements using AI-powered news sentiment analysis and SARIMAX forecasting")
    
    # Initialize components
    data_fetcher = DataFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    predictor = StockPredictor()
    visualizer = Visualizer()
    
    # Sidebar for input
    st.sidebar.header("Configuration")
    
    # Ticker input
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL", help="e.g., AAPL, GOOGL, EUR=X")
    
    # Period selection
    period = st.sidebar.selectbox(
        "Historical Data Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )
    
    # Forecast period
    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to forecast using SARIMAX"
    )
    
    # Analyze button
    if st.sidebar.button("🔍 Analyze", type="primary"):
        if ticker:
            # Clear cache when analyzing a new ticker
            if not hasattr(st.session_state, 'last_ticker') or st.session_state.last_ticker != ticker:
                sentiment_analyzer.clear_cache()
                predictor.clear_cache()
                st.session_state.last_ticker = ticker
                st.session_state.cached_data = None
                st.session_state.cached_news = None
            
            # Cache data fetching
            if st.session_state.get('cached_data') is None:
                with st.spinner("Fetching stock data..."):
                    stock_data, stock_info = data_fetcher.get_stock_data(ticker, period)
                    st.session_state.cached_data = (stock_data, stock_info)
            else:
                stock_data, stock_info = st.session_state.cached_data
            
            if st.session_state.get('cached_news') is None:
                with st.spinner("Fetching news data..."):
                    news_data = data_fetcher.get_news_data(ticker)
                    st.session_state.cached_news = news_data
            else:
                news_data = st.session_state.cached_news
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📰 News Sentiment", "📈 Charts", "🔮 Prediction"])
            
            with tab1:
                display_analysis_tab(stock_data, stock_info, ticker)
            
            with tab2:
                display_sentiment_tab(news_data, sentiment_analyzer, ticker)
            
            with tab3:
                display_charts_tab(stock_data, ticker, visualizer)
            
            with tab4:
                display_prediction_tab(stock_data, news_data, sentiment_analyzer, predictor, ticker, forecast_days)
        
        else:
            st.error("Please enter a ticker symbol.")
    
    # Instructions
    display_sidebar_instructions()

def display_analysis_tab(stock_data, stock_info, ticker):
    st.header(f"Analysis for {ticker.upper()}")
    
    if stock_data is not None and stock_info:
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
        
        with col2:
            change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
            st.metric("Daily Change", f"${change:.2f}", f"{change:.2f}")
        
        with col3:
            volume = stock_data['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
        
        with col4:
            market_cap = stock_info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
            else:
                st.metric("Market Cap", "N/A")
        
        # Company info
        st.subheader("Company Information")
        st.write(f"**Name:** {stock_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
    
    else:
        st.error("Could not fetch stock data. Please check the ticker symbol.")

def display_sentiment_tab(news_data, sentiment_analyzer, ticker):
    st.header("📰 News Sentiment Analysis")
    
    if news_data:
        st.write(f"Found {len(news_data)} news articles. Analyzing top 5...")
        
        # Analyze sentiment
        sentiments = sentiment_analyzer.analyze_news_sentiment(news_data, ticker)
        
        if sentiments:
            # Calculate overall sentiment
            overall_score, overall_label = sentiment_analyzer.calculate_overall_sentiment(sentiments, ticker)
            
            # Display overall sentiment
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Overall Sentiment", overall_label)
            
            with col2:
                st.metric("Sentiment Score", f"{overall_score:.1f}")
            
            # Display individual article sentiments
            st.subheader("Individual Article Analysis")
            
            for i, sentiment in enumerate(sentiments, 1):
                with st.expander(f"Article {i}: {sentiment['title'][:60]}..."):
                    # Display article title and link prominently
                    st.markdown(f"### 📰 {sentiment['title']}")
                    if sentiment['url']:
                        st.markdown(f"🔗 [Read Full Article]({sentiment['url']})")
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Sentiment:** {sentiment['sentiment']}")
                        st.write(f"**Confidence:** {sentiment['confidence']}%")
                        st.write(f"**Market Impact:** {sentiment['market_impact']}")
                    
                    with col2:
                        st.write(f"**Key Factors:**")
                        for factor in sentiment['key_factors']:
                            st.write(f"• {factor}")
                    
                    st.write(f"**Reasoning:** {sentiment['reasoning']}")
        else:
            st.warning("Could not analyze news sentiment. Please check if Ollama is running.")
    else:
        st.warning("No news data found for this ticker.")

def display_charts_tab(stock_data, ticker, visualizer):
    st.header("📈 Price Chart")
    
    if stock_data is not None:
        # Create candlestick chart
        fig = visualizer.create_candlestick_chart(stock_data, ticker)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No stock data available for charting.")

def display_prediction_tab(stock_data, news_data, sentiment_analyzer, predictor, ticker, forecast_days):
    st.header("🔮 Prediction & Forecast")
    
    if stock_data is not None and news_data:
        # Get cached sentiment analysis results
        if sentiment_analyzer.cached_sentiments is not None and sentiment_analyzer.cached_ticker == ticker:
            sentiments = sentiment_analyzer.cached_sentiments
            overall_score, overall_label = sentiment_analyzer.cached_overall_sentiment
        else:
            # This should not happen if tab 2 was visited first, but just in case
            sentiments = sentiment_analyzer.analyze_news_sentiment(news_data, ticker)
            overall_score, overall_label = sentiment_analyzer.calculate_overall_sentiment(sentiments, ticker)
        
        if sentiments:
            # Technical prediction
            technical_prediction, technical_score = predictor.make_technical_prediction(stock_data, overall_score)
            
            # SARIMAX forecast
            with st.spinner("Generating SARIMAX forecast..."):
                forecast_data = predictor.make_sarimax_forecast(stock_data, forecast_days)
            
            # Display predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Technical Prediction", technical_prediction)
            
            with col2:
                st.metric("Technical Score", f"{technical_score:.1f}")
            
            with col3:
                if forecast_data is not None:
                    trend = "Bullish" if forecast_data['forecast'][-1] > stock_data['Close'].iloc[-1] else "Bearish"
                    st.metric("SARIMAX Trend", trend)
                else:
                    st.metric("SARIMAX Trend", "N/A")
            
            # Display forecast chart
            if forecast_data is not None:
                st.subheader("SARIMAX Forecast")
                
                # Create forecast visualization
                visualizer = Visualizer()
                forecast_fig = visualizer.create_forecast_chart(stock_data, forecast_data, ticker)
                print(forecast_fig)
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Display forecast values
                st.subheader("Forecast Values")
                forecast_df = pd.DataFrame({
                    'Date': forecast_data['dates'],
                    'Forecast': forecast_data['forecast'],
                    'Lower CI': forecast_data['lower_ci'],
                    'Upper CI': forecast_data['upper_ci']
                })
                st.dataframe(forecast_df)
            
            # Prediction explanation
            st.subheader("Prediction Factors")
            
            # Technical factors
            current_price = stock_data['Close'].iloc[-1]
            ma_5 = stock_data['Close'].rolling(window=5).mean().iloc[-1]
            ma_20 = stock_data['Close'].rolling(window=20).mean().iloc[-1]
            
            st.write("**Technical Analysis:**")
            st.write(f"• Current Price: ${current_price:.2f}")
            st.write(f"• 5-day MA: ${ma_5:.2f}")
            st.write(f"• 20-day MA: ${ma_20:.2f}")
            
            if current_price > ma_5 > ma_20:
                st.write("• Technical Signal: Bullish (Price above moving averages)")
            elif current_price < ma_5 < ma_20:
                st.write("• Technical Signal: Bearish (Price below moving averages)")
            else:
                st.write("• Technical Signal: Neutral (Mixed signals)")
            
            st.write(f"**Sentiment Analysis:**")
            st.write(f"• Overall Sentiment: {overall_label}")
            st.write(f"• Sentiment Score: {overall_score:.1f}")
            
            # Disclaimer
            st.warning("⚠️ **Disclaimer:** This prediction is for educational purposes only and should not be used as investment advice. Always do your own research and consult with financial professionals before making investment decisions.")
        
        else:
            st.error("Could not generate prediction without sentiment analysis.")
    
    else:
        st.error("Insufficient data for prediction.")

def display_sidebar_instructions():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("""
    1. Make sure Ollama is running locally with Llama3 model
    2. Enter a valid ticker symbol (e.g., AAPL, GOOGL, EUR=X)
    3. Select the historical data period
    4. Set forecast days for SARIMAX
    5. Click 'Analyze' to get predictions
    
    **Requirements:**
    - Ollama running on localhost:11434
    - Llama3 model installed (`ollama pull llama3`)
    
    **New Features:**
    - SARIMAX time series forecasting
    - Modular code structure
    - Improved caching system
    """)

if __name__ == "__main__":
    main()