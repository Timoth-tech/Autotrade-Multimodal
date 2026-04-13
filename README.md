# 🚀 Multimodal Trading: AI Hedge Fund Agent

An autonomous and intelligent trading platform leveraging decision-making agents (LangGraph), sentiment analysis (FinBERT), and advanced quantitative strategies.

![Visual Dashboard Placeholder](https://via.placeholder.com/800x400.png?text=Multimodal+Trading+Dashboard)

## 🌟 Key Features

- **Decision-Making Agents (LangGraph)**: Multi-agent architecture featuring a "Scout", a "Bear" (Pessimist), a "Bull" (Optimist), and a final "Judge".
- **Sentiment Analysis (FinBERT)**: Real-time analysis of financial news to evaluate the psychological impact on assets.
- **Micro & Macro Analysis**: 6-month trend monitoring coupled with intraday technical analysis (RSI, MACD, ATR).
- **Pro Risk Management**: Dynamic **Trailing Stops** calculated based on volatility (ATR).
- **Intraday Panic Monitor**: Security monitor that scans positions every 15 minutes to protect capital.
- **Time-Travel Backtesting**: Ability to simulate agent decisions on past dates.
- **Modern Dashboard**: Next.js interface to visualize positions, news, and the agents' internal debates.

## 🏗️ Architecture

The system relies on an asynchronous autonomous loop:
1. **Scout**: Identifies promising tickers via web search.
2. **Technical Analyst**: Calculates indicators (RSI, ATR, MACD) via `pandas-ta`.
3. **The Debate**: Bull vs. Bear arguments powered by the LLM (Google Gemini).
4. **The Judge**: Final arbitrage based on macro vision and contextual memory (ChromaDB).
5. **Execution**: Automatic order placement via the Alpaca Markets API.

## 🛠️ Quick Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- API Keys: Alpaca Markets (Paper) and Google Gemini.

### Automated Installation
Simply run the setup script at the root:
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Configuration
1. **Backend**:
   - `cd backend`
   - `python -m venv venv`
   - `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
   - `pip install -r requirements.txt`
   - Create your `.env` file from `.env.example`.

2. **Frontend**:
   - `cd frontend`
   - `npm install`
   - `npm run dev`

## 🚀 Usage

1. Launch the Backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
2. Access the Dashboard: `http://localhost:3000`
3. To test a past strategy:
   ```bash
   cd backend
   python backtester.py --symbol AAPL --date 2024-03-15
   ```

## ⚠️ Disclaimer
This software is intended for educational purposes and Paper Trading (Simulation) only. Never use it with real money without being fully aware of the risks associated with derivatives and stock trading.

---
*Developed with ❤️ for smarter trading.*
