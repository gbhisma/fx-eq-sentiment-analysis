import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class StockPredictor:
    """Handles stock prediction using technical analysis and SARIMAX forecasting"""
    
    def __init__(self):
        self.cached_forecasts = {}
    
    def clear_cache(self):
        """Clear cached results"""
        self.cached_forecasts = {}
    
    def make_technical_prediction(self, stock_data, sentiment_score):
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
    
    def make_sarimax_forecast(self, stock_data, forecast_days):
        """Make SARIMAX forecast"""
        try:
            # Try to import statsmodels
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        except ImportError:
            st.error("Please install statsmodels: pip install statsmodels")
            return None
        
        if stock_data is None or len(stock_data) < 30:
            st.error("Insufficient data for SARIMAX forecasting (minimum 30 observations required)")
            return None
        
        try:
            # Cache key
            cache_key = f"{len(stock_data)}_{forecast_days}_{stock_data['Close'].iloc[-1]}"
            if cache_key in self.cached_forecasts:
                return self.cached_forecasts[cache_key]
            
            # Prepare data
            prices = stock_data['Close'].dropna()
            
            # Check for stationarity and apply differencing if needed
            adf_result = adfuller(prices)
            if adf_result[1] > 0.05:  # p-value > 0.05, non-stationary
                prices_diff = prices.diff().dropna()
                d = 1
            else:
                prices_diff = prices
                d = 0
            
            # Simple SARIMAX model with automatic parameter selection
            # Using a basic configuration that works well for most financial time series
            try:
                # Try SARIMAX(1,1,1) first
                model = SARIMAX(prices, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
                fitted_model = model.fit(disp=False)
            except:
                try:
                    # Fallback to simpler model
                    model = SARIMAX(prices, order=(1, 1, 1))
                    fitted_model = model.fit(disp=False)
                except:
                    # Last resort - ARIMA(1,1,1)
                    model = SARIMAX(prices, order=(1, 1, 1))
                    fitted_model = model.fit(disp=False)
            
            # Make forecast
            forecast_result = fitted_model.forecast(steps=forecast_days)
            conf_int = fitted_model.get_forecast(steps=forecast_days).conf_int()
            
            # Generate forecast dates
            last_date = stock_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
            
            # Prepare forecast data
            forecast_data = {
                'dates': forecast_dates,
                'forecast': forecast_result.values,
                'lower_ci': conf_int.iloc[:, 0].values,
                'upper_ci': conf_int.iloc[:, 1].values,
                'model_summary': str(fitted_model.summary())
            }
            
            # Cache the result
            self.cached_forecasts[cache_key] = forecast_data
            
            return forecast_data
            
        except Exception as e:
            st.error(f"Error in SARIMAX forecasting: {e}")
            return None
    
    def calculate_technical_indicators(self, stock_data):
        """Calculate various technical indicators"""
        if stock_data is None or stock_data.empty:
            return {}
        
        indicators = {}
        
        # Moving averages
        indicators['MA_5'] = stock_data['Close'].rolling(window=5).mean()
        indicators['MA_10'] = stock_data['Close'].rolling(window=10).mean()
        indicators['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        
        # RSI
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = exp1 - exp2
        indicators['MACD_signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        rolling_mean = stock_data['Close'].rolling(window=20).mean()
        rolling_std = stock_data['Close'].rolling(window=20).std()
        indicators['BB_upper'] = rolling_mean + (rolling_std * 2)
        indicators['BB_lower'] = rolling_mean - (rolling_std * 2)
        
        return indicators