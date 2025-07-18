import yfinance as yf
import streamlit as st

class DataFetcher:
    """Handles fetching stock data and news from yfinance"""
    
    def __init__(self):
        pass
    
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