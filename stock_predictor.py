import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Stock/Forex Predictor",
    page_icon="📈",
    layout="wide"
)

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using Ollama Llama3"""
        try:
            prompt = f"""
            Analyze the sentiment of the following financial news text and provide:
            1. Overall sentiment (Positive, Negative, or Neutral)
            2. Confidence score (0-100)
            3. Key factors influencing the sentiment
            4. Potential market impact (Bullish, Bearish, or Neutral)
            
            Text: {text}
            
            Please respond in JSON format:
            {{
                "sentiment": "Positive/Negative/Neutral",
                "confidence": 85,
                "key_factors": ["factor1", "factor2"],
                "market_impact": "Bullish/Bearish/Neutral",
                "reasoning": "brief explanation"
            }}
            """
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                try:
                    # Extract JSON from response
                    response_text = result['response']
                    # Find JSON in the response
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                except:
                    # Fallback if JSON parsing fails
                    return {
                        "sentiment": "Neutral",
                        "confidence": 50,
                        "key_factors": ["Analysis failed"],
                        "market_impact": "Neutral",
                        "reasoning": "Could not parse AI response"
                    }
            else:
                return None
        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}")
            return None

class StockPredictor:
    def __init__(self):
        self.ollama = OllamaClient()
    
    def get_stock_data(self, ticker, period="1mo"):
        """Get stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            info = stock.info
            return hist, info
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return None, None
    
    def get_news_data(self, ticker):
        """Get news data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            return news
        except Exception as e:
            st.error(f"Error fetching news data: {e}")
            return []
    
    def analyze_news_sentiment(self, news_data):
        """Analyze sentiment of news articles"""
        sentiments = []
        
        for article in news_data[:5]:  # Analyze top 5 news articles
            title = article.get("content").get('title', '')
            summary = article.get("content").get('summary', '')
            
            # Combine title and summary for analysis
            text = f"{title}. {summary}"
            
            with st.spinner(f"Analyzing: {title[:50]}..."):
                sentiment = self.ollama.analyze_sentiment(text)
                
                if sentiment:
                    sentiment['title'] = title
                    sentiment['url'] = article.get("content").get("canonicalUrl").get('url', '')
                    sentiment['published'] = article.get("content").get('pubDate', 0)
                    sentiments.append(sentiment)
                
                # Add small delay to avoid overwhelming the API
                time.sleep(1)
        
        return sentiments
    
    def calculate_overall_sentiment(self, sentiments):
        """Calculate overall sentiment score"""
        if not sentiments:
            return 0, "Neutral"
        
        total_score = 0
        total_weight = 0
        
        for sentiment in sentiments:
            confidence = sentiment['confidence']
            
            if sentiment['sentiment'] == 'Positive':
                score = confidence
            elif sentiment['sentiment'] == 'Negative':
                score = -confidence
            else:
                score = 0
            
            total_score += score
            total_weight += confidence
        
        if total_weight == 0:
            return 0, "Neutral"
        
        overall_score = total_score / total_weight
        
        if overall_score > 20:
            return overall_score, "Positive"
        elif overall_score < -20:
            return overall_score, "Negative"
        else:
            return overall_score, "Neutral"
    
    def make_prediction(self, stock_data, sentiment_score, sentiment_label):
        """Make prediction based on technical and sentiment analysis"""
        if stock_data is None or stock_data.empty:
            return "Insufficient data", 0
        
        # Calculate technical indicators
        current_price = stock_data['Close'].iloc[-1]
        ma_5 = stock_data['Close'].rolling(window=5).mean().iloc[-1]
        ma_20 = stock_data['Close'].rolling(window=20).mean().iloc[-1]
        
        # Price momentum
        price_change = (current_price - stock_data['Close'].iloc[-5]) / stock_data['Close'].iloc[-5] * 100
        
        # Technical signal
        technical_signal = 0
        if current_price > ma_5 > ma_20:
            technical_signal = 1
        elif current_price < ma_5 < ma_20:
            technical_signal = -1
        
        # Combine technical and sentiment
        prediction_score = (technical_signal * 40) + (sentiment_score * 0.6)
        
        if prediction_score > 15:
            prediction = "Bullish"
        elif prediction_score < -15:
            prediction = "Bearish"
        else:
            prediction = "Neutral"
        
        return prediction, prediction_score

def main():
    st.title("📈 Stock/Forex Predictor with News Sentiment Analysis")
    st.markdown("### Predict market movements using AI-powered news sentiment analysis")
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Sidebar for input
    st.sidebar.header("Configuration")
    
    # Ticker input
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL", help="e.g., AAPL, GOOGL, EUR=X")
    
    # Period selection
    period = st.sidebar.selectbox(
        "Historical Data Period",
        ["1mo", "3mo", "6mo", "1y"],
        index=0
    )
    
    # Analyze button
    if st.sidebar.button("🔍 Analyze", type="primary"):
        if ticker:
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📰 News Sentiment", "📈 Price Chart", "🔮 Prediction"])
            
            with tab1:
                st.header(f"Analysis for {ticker.upper()}")
                
                # Get stock data
                with st.spinner("Fetching stock data..."):
                    stock_data, stock_info = predictor.get_stock_data(ticker, period)
                
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
            
            with tab2:
                st.header("📰 News Sentiment Analysis")
                
                # Get news data
                with st.spinner("Fetching news data..."):
                    news_data = predictor.get_news_data(ticker)
                
                if news_data:
                    st.write(f"Found {len(news_data)} news articles. Analyzing top 5...")
                    
                    # Analyze sentiment
                    sentiments = predictor.analyze_news_sentiment(news_data)
                    
                    if sentiments:
                        # Calculate overall sentiment
                        overall_score, overall_label = predictor.calculate_overall_sentiment(sentiments)
                        
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
            
            with tab3:
                st.header("📈 Price Chart")
                
                if stock_data is not None:
                    # Create candlestick chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Price', 'Volume'),
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.1
                    )
                    
                    # Candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=stock_data.index,
                            open=stock_data['Open'],
                            high=stock_data['High'],
                            low=stock_data['Low'],
                            close=stock_data['Close'],
                            name='Price'
                        ),
                        row=1, col=1
                    )
                    
                    # Moving averages
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'].rolling(window=5).mean(),
                            name='MA 5',
                            line=dict(color='orange', width=1)
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'].rolling(window=20).mean(),
                            name='MA 20',
                            line=dict(color='red', width=1)
                        ),
                        row=1, col=1
                    )
                    
                    # Volume chart
                    fig.add_trace(
                        go.Bar(
                            x=stock_data.index,
                            y=stock_data['Volume'],
                            name='Volume',
                            marker_color='lightblue'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title=f"{ticker.upper()} Stock Price and Volume",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error("No stock data available for charting.")
            
            with tab4:
                st.header("🔮 Prediction")
                
                if stock_data is not None and news_data:
                    # Get sentiment analysis results
                    sentiments = predictor.analyze_news_sentiment(news_data)
                    
                    if sentiments:
                        overall_score, overall_label = predictor.calculate_overall_sentiment(sentiments)
                        
                        # Make prediction
                        prediction, prediction_score = predictor.make_prediction(
                            stock_data, overall_score, overall_label
                        )
                        
                        # Display prediction
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Prediction", prediction)
                        
                        with col2:
                            st.metric("Prediction Score", f"{prediction_score:.1f}")
                        
                        with col3:
                            st.metric("Confidence", "Medium")
                        
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
        
        else:
            st.error("Please enter a ticker symbol.")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("""
    1. Make sure Ollama is running locally with Llama3 model
    2. Enter a valid ticker symbol (e.g., AAPL, GOOGL, EUR=X)
    3. Select the historical data period
    4. Click 'Analyze' to get predictions
    
    **Requirements:**
    - Ollama running on localhost:11434
    - Llama3 model installed (`ollama pull llama3`)
    """)

if __name__ == "__main__":
    main()