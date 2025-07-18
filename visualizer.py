import plotly.graph_objects as go
import pandas as pd

class Visualizer:
    def create_forecast_chart(self, stock_data, forecast_data, ticker):
        """
        Create a Plotly chart for forecasted prices with confidence intervals.
        """
        # Extract last actual date
        last_actual_date = stock_data.index[-1]

        # Combine actual and forecasted data
        actual_dates = stock_data.index
        actual_prices = stock_data['Close']

        forecast_dates = forecast_data['dates']
        forecast_prices = forecast_data['forecast']
        lower_ci = forecast_data['lower_ci']
        upper_ci = forecast_data['upper_ci']

        # Create figure
        fig = go.Figure()

        # Actual price line
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_prices,
            mode='lines',
            name='Actual Price',
            line=dict(color='blue')
        ))

        # Forecasted price line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_prices,
            mode='lines',
            name='Forecasted Price',
            line=dict(color='green', dash='dash')
        ))

        # Confidence interval (shaded area)
        fig.add_trace(go.Scatter(
            x=forecast_dates.tolist() + forecast_dates[::-1].tolist(),
            y=upper_ci.tolist() + lower_ci[::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

        fig.update_layout(
            title=f"{ticker} Forecast with Confidence Interval",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            legend=dict(x=0, y=1.1, orientation="h")
        )

        return fig

    def create_candlestick_chart(self, stock_data, ticker):
        """
        Create a candlestick chart of stock data
        """
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])

        fig.update_layout(
            title=f"{ticker} Price Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )

        return fig
