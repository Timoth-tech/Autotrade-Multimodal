import argparse
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from unittest.mock import patch
import logging

# Mute third-party logs for clarity
logging.getLogger("ddgs").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

def run_historical_backtest(symbol: str, target_date_str: str):
    """
    Launches the LangGraph agent in the past by mocking the date.
    Note: DDG results may slightly diverge as it is a live search API.
    """
    print(f"\n=============================================")
    print(f"🚀 LAUNCHING BACKTESTING - TIME MACHINE")
    print(f"Ticker : {symbol}")
    print(f"Date of return : {target_date_str}")
    print(f"=============================================\n")

    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        print("Error: Date format must be YYYY-MM-DD.")
        sys.exit(1)

    load_dotenv()
    
    # 1. Monkey-Patch datetime.now() to fool the Alpaca agent / technical indicators
    class MockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target_date

    import datetime as real_datetime
    real_datetime.datetime = MockDatetime

    print("✅ Internal system date spoofed.")
    print("⏳ Starting deliberation pipeline (Bull/Bear/Judge)...\n")
    
    try:
        from agent import trading_agent
        
        # Run the agent
        state = trading_agent.invoke({"symbol": symbol})
        
        print("\n=============================================")
        print("📈 BACKTEST RESULTS")
        print("=============================================")
        print(f"Symbol         : {state.get('symbol')}")
        print(f"Judge Decision : {state.get('final_decision')}")
        print(f"Volatility (ATR): Extraction via Pandas-TA calculated at {target_date_str}")
        print(f"Trailing Stop  : {state.get('trail_percent', 0.0):.2f}%")
        print("\nDeliberation Messages:")
        for msg in state.get('messages', []):
             if "Judge's Verdict" in msg or "Thesis" in msg:
                 print(f" > {msg}")
                 
        print("\n(Tip: You can compare this decision with the actual price curve of the stock in the days that followed!)")
        
    except Exception as e:
         print(f"\n❌ Error during backtesting: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotrade AI - Time Backtester")
    parser.add_argument("--symbol", type=str, required=True, help="The stock ticker (e.g., AAPL, TSLA)")
    parser.add_argument("--date", type=str, required=True, help="The past date (Format: YYYY-MM-DD)")
    
    args = parser.parse_args()
    run_historical_backtest(args.symbol, args.date)
