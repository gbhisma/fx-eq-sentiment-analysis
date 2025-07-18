# 📊 News-Driven Stock & Forex Predictor

This is a beginner-friendly, AI-powered application for predicting stock and forex price movements using real-time news sentiment analysis and time-series forecasting. It uses:

* 📈 **Historical price data** via `yfinance`
* 🧠 **Sentiment analysis** via [Ollama](https://ollama.com/) (Llama3 or DeepSeek)
* 🔮 **Forecasting** using SARIMAX time series models
* 🌐 **Web interface** via Streamlit

---

## 🚀 Features

* ✅ Real-time news sentiment using LLM (Llama3 / DeepSeek)
* ✅ Technical analysis with moving averages
* ✅ SARIMAX forecasting with margin of error
* ✅ Streamlit UI with charts, metrics, and news explanations

---

## 🖥️ System Requirements

* Python 3.9 or later
* Internet connection
* [Ollama installed locally](https://ollama.com/download)

---

## 📦 Installation Guide

### 1. 🔧 Install Python (if not already)

#### Windows:

* Download from [python.org/downloads](https://www.python.org/downloads/windows/)
* During installation, **check** "Add Python to PATH"

#### Mac/Linux:

* Mac: Use [Homebrew](https://brew.sh): `brew install python`
* Linux (Debian/Ubuntu): `sudo apt install python3 python3-pip`

---

### 2. 📂 Clone or Download This Repository

```bash
# Via Git (recommended)
git clone https://github.com/yourusername/news-stock-predictor.git
cd news-stock-predictor

# Or download ZIP and extract manually
```

---

### 3. 🧪 Set Up Virtual Environment (Optional but Recommended)

```bash
# Windows\python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

---

### 4. 📥 Install Required Python Packages

```bash
pip install -r requirements.txt
```

---

### 5. 🤖 Install and Run Ollama

Download Ollama from: [https://ollama.com/download](https://ollama.com/download)

Then, in terminal:

```bash
ollama run llama3
```

Make sure it's running locally at `http://localhost:11434`

---

### 6. ▶️ Run the App

```bash
streamlit run main.py
```

It will open in your browser (usually at [http://localhost:8501](http://localhost:8501))

---

## 📚 How to Use

1. Enter a stock/forex symbol (e.g. `AAPL`, `GOOGL`, or `EUR=X`)
2. Choose historical range (e.g. 3 months)
3. Set how many days to forecast (e.g. 7)
4. Click **Analyze**
5. Review the sentiment, technical indicators, and forecast chart

> Example forex symbols:
>
> * USD/EUR → `EUR=X`
> * USD/JPY → `JPY=X`

---

## 🧠 Model Notes

* **News Sentiment**: Processed using an LLM (Llama3 or DeepSeek) locally
* **Forecasting**: Uses SARIMAX with rolling confidence intervals
* **No financial advice**: This is an educational tool!

---

## 🛠️ Troubleshooting

### Ollama isn't working?

* Ensure you ran `ollama run llama3`
* Restart your terminal or system
* Make sure Ollama listens at `http://localhost:11434`

### Streamlit isn't opening?

* Try accessing [http://localhost:8501](http://localhost:8501) manually
* Or run `streamlit run main.py`

### Missing packages?

* Re-run: `pip install -r requirements.txt`

---

Happy Forecasting! 📈✨
