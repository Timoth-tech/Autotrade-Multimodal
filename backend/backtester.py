import argparse
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from unittest.mock import patch
import logging

# Mettre en sourdine les logs tiers pour la clarté
logging.getLogger("ddgs").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

def run_historical_backtest(symbol: str, target_date_str: str):
    """
    Lance l'agent LangGraph dans le passé en mockant la date.
    Note: Les résultats DDG peuvent légèrement diverger car c'est une API de recherche live.
    """
    print(f"\n=============================================")
    print(f"🚀 LANCEMENT DU BACKTESTING - MACHINE À VOYAGER DANS LE TEMPS")
    print(f"Ticker : {symbol}")
    print(f"Date de retour : {target_date_str}")
    print(f"=============================================\n")

    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        print("Erreur: Le format de la date doit être YYYY-MM-DD.")
        sys.exit(1)

    load_dotenv()
    
    # 1. On "Monkey-Patch" datetime.now() pour tromper l'agent Alpaca / indicateurs techniques
    class MockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return target_date

    import datetime as real_datetime
    real_datetime.datetime = MockDatetime

    print("✅ Date système interne falsifiée.")
    print("⏳ Démarrage du pipeline de réflexion (Bull/Bear/Judge)...\n")
    
    try:
        from agent import trading_agent
        
        # On exécute l'agent
        state = trading_agent.invoke({"symbol": symbol})
        
        print("\n=============================================")
        print("📈 RÉSULTATS DU BACKTEST")
        print("=============================================")
        print(f"Symbole        : {state.get('symbol')}")
        print(f"Décision Juge  : {state.get('final_decision')}")
        print(f"Volatilité (ATR): Extraction via Pandas-TA calculée au {target_date_str}")
        print(f"Trailing Stop  : {state.get('trail_percent', 0.0):.2f}%")
        print("\nMessages de Délibération :")
        for msg in state.get('messages', []):
             if "Verdict du Juge" in msg or "Thèse" in msg:
                 print(f" > {msg}")
                 
        print("\n(Astuce : Vous pouvez comparer cette décision avec la courbe de prix réelle du titre les jours qui ont suivi !)")
        
    except Exception as e:
         print(f"\n❌ Erreur durant le backtesting : {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotrade AI - Backtester Temporel")
    parser.add_argument("--symbol", type=str, required=True, help="Le ticker boursier (ex: AAPL, TSLA)")
    parser.add_argument("--date", type=str, required=True, help="La date passée (Format: YYYY-MM-DD)")
    
    args = parser.parse_args()
    run_historical_backtest(args.symbol, args.date)
