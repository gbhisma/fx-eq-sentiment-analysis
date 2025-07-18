import requests
import json
import streamlit as st
import time

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def analyze_sentiment(self, text, ticker):
        """Analyze sentiment using Ollama Llama3"""
        try:
            prompt = f"""
            Analyze the sentiment of the following financial news text and provide:
            1. Overall sentiment (Positive, Negative, or Neutral) toward this ticker: {ticker}
            2. Confidence score (0-100)
            3. Key factors influencing the sentiment
            4. Potential market impact (Bullish, Bearish, or Neutral) toward {ticker}
            
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

class SentimentAnalyzer:
    """Handles news sentiment analysis with caching"""
    
    def __init__(self):
        self.ollama = OllamaClient()
        # Cache for analysis results
        self.cached_sentiments = None
        self.cached_ticker = None
        self.cached_overall_sentiment = None
    
    def clear_cache(self):
        """Clear cached results"""
        self.cached_sentiments = None
        self.cached_ticker = None
        self.cached_overall_sentiment = None
    
    def analyze_news_sentiment(self, news_data, ticker):
        """Analyze sentiment of news articles with caching"""
        # Check if we already have results for this ticker
        if self.cached_sentiments is not None and self.cached_ticker == ticker:
            return self.cached_sentiments
        
        sentiments = []
        
        for article in news_data[:5]:  # Analyze top 5 news articles
            title = article.get("content").get('title', '')
            summary = article.get("content").get('summary', '')
            
            # Combine title and summary for analysis
            text = f"{title}. {summary}"
            
            with st.spinner(f"Analyzing: {title[:50]}..."):
                sentiment = self.ollama.analyze_sentiment(text, ticker)
                
                if sentiment:
                    sentiment['title'] = title
                    sentiment['url'] = article.get("content").get("canonicalUrl").get('url', '')
                    sentiment['published'] = article.get("content").get('pubDate', 0)
                    sentiments.append(sentiment)
                
                # Add small delay to avoid overwhelming the API
                time.sleep(1)
        
        # Cache the results
        self.cached_sentiments = sentiments
        self.cached_ticker = ticker
        
        return sentiments
    
    def calculate_overall_sentiment(self, sentiments, ticker):
        """Calculate overall sentiment score with caching"""
        # Check if we already have results for this ticker
        if self.cached_overall_sentiment is not None and self.cached_ticker == ticker:
            return self.cached_overall_sentiment
        
        if not sentiments:
            result = (0, "Neutral")
        else:
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
                result = (0, "Neutral")
            else:
                overall_score = total_score / total_weight
                
                if overall_score > 20:
                    result = (overall_score, "Positive")
                elif overall_score < -20:
                    result = (overall_score, "Negative")
                else:
                    result = (overall_score, "Neutral")
        
        # Cache the result
        self.cached_overall_sentiment = result
        return result