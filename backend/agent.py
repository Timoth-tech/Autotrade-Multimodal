import os
import re
import math
import time
import logging
import operator
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated, Sequence
from langchain_community.tools import DuckDuckGoSearchResults
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from news_agent import news_pipeline_agent, LATEST_FINBERT_NEWS

# --- Third-party Logging Optimization ---
logging.getLogger("ddgs").setLevel(logging.ERROR)
logging.getLogger("primp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

import json

GLOBAL_DEBATES_CACHE = []

def safe_ddg_search(query: str, max_retries: int = 3) -> str:
    """Resilient wrapper for DuckDuckGo. Retry with backoff on network errors."""
    search = DuckDuckGoSearchResults()
    for attempt in range(max_retries):
        try:
            result = search.invoke(query)
            if result:
                # Return raw text result for the LLM
                return result
        except Exception as e:
            logger.warning(f"[DDG] Attempt {attempt+1}/{max_retries} failed for '{query[:50]}...' : {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # backoff: 1s, 2s, 4s
    return f"Search unavailable for '{query}' after {max_retries} attempts."

# --- Structured Logging (#8) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("trading.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("autotrade")

# --- Removed localized FinBERT pipeline to use the one in news_agent.py ---
# This is the state our agent components will pass to each other
class CandidateAnalysis(TypedDict, total=False):
    symbol: str
    text_analysis: str
    bull_argument: str
    bear_argument: str
    conviction_score: int

class AgentState(TypedDict):
    symbol: str
    text_analysis: str
    bull_argument: str
    bear_argument: str
    final_decision: str
    trade_notional: float
    trade_qty: float
    tp_price: float
    sl_price: float
    trail_percent: float
    danger_score: int
    candidates: list  # (#13) Liste de CandidateAnalysis
    messages: Annotated[Sequence[str], operator.add]

# --- Appel LLM avec retry (#7) ---
@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
def safe_llm_call(prompt, llm=None):
    """Appelle le LLM avec retry exponentiel en cas d'erreur."""
    if llm is None:
        llm = get_llm()
    return get_content(llm.invoke(prompt))

def get_llm(model_name="gemma-4-31b-it"):
    """Initializes and returns the Gemma/Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "votre_cle_api_google_gemini":
        logger.warning("GOOGLE_API_KEY missing or placeholder. Requests will fail.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        max_output_tokens=1000
    )

def get_vector_db():
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import os
    
    api_key = os.getenv("GOOGLE_API_KEY")
    # Using the correct embedding model name from the futuristic API
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    vectorstore = Chroma(
        collection_name="trading_memory",
        embedding_function=embeddings,
        persist_directory="./chroma_data"
    )
    return vectorstore

def get_content(response):
    """Helper to extract text from LLM response (supports Gemma-4 thinking format)."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "").strip()
    return str(content).strip()


def analyze_portfolio_logic():
    """Fetches Alpaca portfolio and uses Gemini to analyze performance and provide insights."""
    from alpaca.trading.client import TradingClient
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or api_key == "votre_cle_api_alpaca":
            return "Veuillez configurer vos clés Alpaca dans .env pour l'analyse."
            
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()
        
        pos_str = "\n".join([f"- {p.symbol}: {p.qty} shares at ${p.current_price} (Unrealized P&L: {p.unrealized_pl}$)" for p in positions])
        if not pos_str:
            pos_str = "No open positions."
            
        context = f"Total balance: ${account.equity}\nAvailable cash: ${account.cash}\nOpen positions:\n{pos_str}"
        prompt = f"You are a financial expert. Here is the current state of my autonomous trading portfolio:\n{context}\n\nWrite a short analysis (3-4 sentences in English) of these results. Congratulate or warn based on P&L, and give quick advice on exposure."
        
        try:
            llm = get_llm()
            return get_content(llm.invoke(prompt))
        except Exception as e:
            return f"Erreur Gemini lors de l'analyse : {str(e)}"
    except Exception as e:
        return f"Erreur de récupération Alpaca : {str(e)}"


def update_strategy_from_performance():
    """Analyse les gains/pertes du portefeuille, track les résultats réels (#5), et met à jour les règles dans ChromaDB."""
    from alpaca.trading.client import TradingClient
    logger.info("[REFLECTION] Mise à jour de la stratégie en fonction des performances...")
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or api_key == "votre_cle_api_alpaca":
            return
            
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()
        
        # --- Track résultats réels (#5) ---
        from langchain_core.documents import Document
        db = get_vector_db()
        
        for p in positions:
            pl_pct = float(p.unrealized_plpc) * 100 if p.unrealized_plpc else 0
            status = "Success" if pl_pct > 0 else "Failure"
            track_doc = (
                f"Trade {p.symbol}: Entered at {p.avg_entry_price}$, "
                f"currently {p.current_price}$ ({pl_pct:+.1f}%). {status}. "
                f"Position of {p.qty} shares, Unrealized P&L: {p.unrealized_pl}$."
            )
            doc = Document(page_content=track_doc, metadata={"source": "performance_track", "symbol": p.symbol, "timestamp": time.time()})
            db.add_documents([doc])
        
        logger.info(f"[REFLECTION] {len(positions)} positions trackées dans ChromaDB.")
        
        pos_str = "\n".join([f"- {p.symbol}: {p.qty} shares, Unrealized P&L: {p.unrealized_pl}$ ({float(p.unrealized_plpc)*100 if p.unrealized_plpc else 0:.1f}%)" for p in positions])
        if not pos_str:
            pos_str = "No positions. No history or everything closed."
            
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(period="1M", timeframe="1D")
        history = client.get_portfolio_history(req)
        profit_loss_1m = history.equity[-1] - history.equity[0] if history.equity else 0
        
        context = f"Current portfolio: {account.equity}$. 1-month P&L: {profit_loss_1m}$.\nOpen positions:\n{pos_str}"
        prompt = f"You are the critical mind of a trading algorithm. Here is the portfolio state:\n{context}\n\nIdentify mistakes (e.g., losing positions) and successes. Based on these gains/losses, write 3 strict and concise rules (in English) that you must follow for your next trade (e.g., avoid a sector, cut losses early). Return ONLY the 3 rules as bullet points."
        
        strategy_rules = safe_llm_call(prompt)
        
        # Découpe en règles distinctes pour la sémantique
        rules = [r.strip() for r in strategy_rules.split('\n') if r.strip().startswith('-') or len(r.strip()) > 10]
        if not rules:
            rules = [strategy_rules]
            
        docs = [Document(page_content=r, metadata={"source": "performance_review", "timestamp": time.time()}) for r in rules]
        db.add_documents(docs)
        
        logger.info(f"[REFLECTION] {len(docs)} nouvelles règles insérées dans ChromaDB.")
    except Exception as e:
        logger.error(f"[REFLECTION ERROR] Impossible d'affiner la stratégie : {str(e)}")

def update_macro_strategy():
    """Analyse les tendances macro-économiques et génère une vision à 6 mois."""
    logger.info("[MACRO OUTLOOK] Mise à jour de la vision macro-économique...")
    try:
        # Recherche globale web sur les tendances à venir
        news = safe_ddg_search("global macroeconomic trends stock market outlook next 6 months sectors to pick", max_retries=2)
        
        prompt = (
            f"You are the chief strategist of a top-tier investment fund.\n"
            f"Here are the latest global economic news:\n{news}\n\n"
            f"Write a concise and sharp strategic vision (max 4-5 sentences) that:\n"
            f"1. Summarizes the current macro-economic sentiment.\n"
            f"2. Identifies upcoming challenges and the most resilient/promising sectors for the next 6 months (as well as those to avoid absolutely).\n"
            f"Only provide your analysis, without any politeness formulas."
        )
        
        macro_vision = safe_llm_call(prompt)
        
        db = get_vector_db()
        from langchain_core.documents import Document
        # L'ancienne vision sera noyée, mais on filtrera lors de la recherche
        doc = Document(
            page_content=f"VISION MACRO À 6 MOIS: {macro_vision}", 
            metadata={"source": "macro_outlook", "timestamp": time.time()}
        )
        db.add_documents([doc])
        logger.info("[MACRO OUTLOOK] Nouvelle vision macro à 6 mois sauvegardée dans ChromaDB.")
        
    except Exception as e:
        logger.error(f"[MACRO OUTLOOK] Erreur lors de l'analyse macro : {str(e)}")


def prospector_node(state: AgentState):
    """(#13) Searches the web and picks UP TO 5 promising tickers for multi-analysis."""
    logger.info("[PROSPECTOR NODE] Recherche multi-ticker (Diversification ciblée)...")
    
    try:
        from alpaca.trading.client import TradingClient
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        current_positions_str = "Aucune position actuelle."
        if api_key and api_key != "votre_cle_api_alpaca":
            try:
                client = TradingClient(api_key, secret_key, paper=True)
                positions = client.get_all_positions()
                if positions:
                    current_positions_str = ", ".join([p.symbol for p in positions])
            except Exception as e:
                logger.warning(f"Could not fetch Alpaca positions for prospecting: {e}")

        # Recherche plus globale, moins centrée "tech"
        news = safe_ddg_search("top stocks to buy today promising unvalued different sectors healthcare energy finance tech consumer")
        
        # Règles stratégiques depuis ChromaDB
        saved_strategy = ""
        try:
            db = get_vector_db()
            macro_docs = db.similarity_search("macro outlook", k=1, filter={"source": "macro_outlook"})
            if macro_docs:
                saved_strategy = f"\n\n{macro_docs[0].page_content}"
                
            strategy_docs = db.similarity_search("trading strategy rules mistakes success", k=3)
            if strategy_docs:
                rules = "\n".join([f"- {d.page_content}" for d in strategy_docs])
                saved_strategy += f"\n\nRECENT STRATEGIC RULES:\n{rules}"
        except Exception as e:
            logger.warning(f"ChromaDB unavailable for strategy: {e}")

        prompt = (
            f"You are a ruthless fund manager with over 30 years of experience. Your goal: 20% annual return while GUARANTEEING STRICT SECTOR DIVERSIFICATION.\n\n"
            f"Here are my current positions: {current_positions_str}\n"
            f"Implicitly analyze the sectors of my current holdings (e.g., Tech, Finance, Healthcare).\n"
            f"Based on the news and the 6-MONTH MACRO VISION (below), pick EXACTLY 5 stock tickers that present the best opportunities.\n"
            f"MAJOR CRITERION 1: Diversification. You MUST select stocks from VARIED categories (e.g., if I have a lot of AI/Tech, pick Healthcare, Energy, etc.).\n"
            f"MAJOR CRITERION 2: Long-term trend. Prefer companies adapted to global economic outlooks (rates, inflation).\n\n"
            f"Today's news: {news}\n"
            f"{saved_strategy}\n\n"
            f"For each ticker, write one line in the format: TICKER | Sector | Short justification (1 sentence)\n"
            f"Example:\nJNJ | Healthcare | Strong dividends and protection against tech volatility.\nXOM | Energy | Rising oil prices and very undervalued fundamentals.\n\n"
        )
        
        try:
            response = safe_llm_call(prompt)
            tickers = []
            for line in response.split('\n'):
                line = line.strip()
                if '|' in line:
                    parts = line.split('|', 1)
                    ticker = re.sub(r'[^A-Z]', '', parts[0].strip().upper())
                    reason = parts[1].strip() if len(parts) > 1 else ""
                    if ticker and len(ticker) <= 5:
                        tickers.append({"symbol": ticker, "reason": reason})
            
            # Fallback si le parsing échoue
            if not tickers:
                tickers = [{"symbol": "AAPL", "reason": "Fallback"}]
            
            # Limiter à 5 max
            tickers = tickers[:5]
            
        except Exception as e:
            tickers = [{"symbol": "AAPL", "reason": f"Fallback suite à erreur : {e}"}]
        
        ticker_names = [t['symbol'] for t in tickers]
        logger.info(f"[PROSPECTOR NODE] {len(tickers)} candidats identifiés : {ticker_names}")
        
        return {
            "candidates": tickers,
            "messages": [f"🌐 Prospection terminée. {len(tickers)} candidats : {', '.join(ticker_names)}"]
        }
    except Exception as e:
        logger.error(f"[PROSPECTOR NODE] Erreur : {e}")
        return {
            "candidates": [{"symbol": "AAPL", "reason": "Fallback"}],
            "messages": [f"Error during Web search: {str(e)}"]
        }


def _fetch_fundamentals(symbol: str) -> str:
    """Helper : récupère les données fondamentales via Alpaca et Pandas TA (RSI, MACD, ATR)."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta
        import pandas as pd
        import pandas_ta as ta
        
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        data_client = StockHistoricalDataClient(api_key, secret_key)
        
        latest_req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        latest = data_client.get_stock_latest_trade(latest_req)
        current_price = latest[symbol].price
        
        end = datetime.now()
        start = end - timedelta(days=150)
        bars_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end)
        bars = data_client.get_stock_bars(bars_req)
        bar_list = bars[symbol] if symbol in bars else []
        
        if bar_list:
            df = pd.DataFrame([{
                'date': b.timestamp,
                'open': float(b.open),
                'high': float(b.high),
                'low': float(b.low),
                'close': float(b.close),
                'volume': float(b.volume)
            } for b in bar_list])
            df.set_index('date', inplace=True)
            
            # --- CALCUL DES INDICATEURS VIA PANDAS-TA ---
            # SMA 20 et 50
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            # RSI 14
            df.ta.rsi(length=14, append=True)
            # MACD
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            # ATR 14
            df.ta.atr(length=14, append=True)
            
            # On prend la dernière ligne valide
            last_row = df.iloc[-1]
            
            sma_20 = last_row.get('SMA_20', current_price) if pd.notna(last_row.get('SMA_20')) else current_price
            sma_50 = last_row.get('SMA_50', current_price) if pd.notna(last_row.get('SMA_50')) else current_price
            rsi_14 = last_row.get('RSI_14', 50) if pd.notna(last_row.get('RSI_14')) else 50
            macd_val = last_row.get('MACD_12_26_9', 0) if pd.notna(last_row.get('MACD_12_26_9')) else 0
            atr_14 = last_row.get('ATRr_14', current_price * 0.02) if pd.notna(last_row.get('ATRr_14')) else (current_price * 0.02)
            
            # Formatting and alerts
            signal = "Golden Cross (↑)" if sma_20 > sma_50 else "Death Cross (↓)"
            trend = "↑ BULLISH" if current_price > sma_50 else "↓ BEARISH"
            
            rsi_alert = "OVERBOUGHT (RSI>70)" if rsi_14 > 70 else ("OVERSOLD (RSI<30)" if rsi_14 < 30 else "Neutral")
            macd_signal = "BULLISH (MACD>0)" if macd_val > 0 else "BEARISH (MACD<0)"
            
            return (
                f"\n📊 TECHNICAL DATA [{symbol}] | Current Price: {current_price:.2f}$ \n"
                f"- Trend: {trend} (SMA20: {sma_20:.2f}$, SMA50: {sma_50:.2f}$ → {signal})\n"
                f"- Volatility (ATR): {atr_14:.2f}$ per day (Expected average move).\n"
                f"- Momentum: RSI={rsi_14:.1f} ({rsi_alert}) | MACD: {macd_signal}"
            )
        return f"Historical bars unavailable for {symbol}."
    except Exception as e:
        return f"Fundamentals unavailable for {symbol}: {e}"


def _analyze_single_ticker(symbol: str) -> dict:
    """(#13) Analyse complète d'un ticker : news + fondamentaux + bull + bear + score conviction."""
    logger.info(f"[MULTI-ANALYZER] Analyse de {symbol}...")
    
    # 1. Vraies news via News Agent (Web + X)
    try:
        import asyncio
        news = asyncio.run(news_pipeline_agent.analyze_market_sentiment_combined(symbol))
    except Exception:
        news = "News indisponibles."
    
    # 2. Fondamentaux
    fundamentals = _fetch_fundamentals(symbol)
    text_analysis = f"{news}\n{fundamentals}"
    
    # 3. Bull
    bull_prompt = (
        f"You are a multimillionaire trader with 30 years of experience. Analyzing stock {symbol}.\n"
        f"Data: '{text_analysis[:500]}'.\n"
        f"Find 2 powerful arguments justifying a BUY. Maximum 3 sentences."
    )
    try:
        bull = safe_llm_call(bull_prompt)
    except Exception as e:
        bull = f"Bull Error: {e}"
    
    # 4. Bear
    bear_prompt = (
        f"You are a fearsome short-seller with 30 years of experience. Analyzing stock {symbol}.\n"
        f"Data: '{text_analysis[:500]}'.\n"
        f"Find 2 cynical arguments justifying a SELL. Maximum 3 sentences."
    )
    try:
        bear = safe_llm_call(bear_prompt)
    except Exception as e:
        bear = f"Bear Error: {e}"
    
    # 5. Score de conviction
    conv_prompt = (
        f"CONVICTION score from 1 to 10 for a trade on {symbol}.\n"
        f"Bull: {bull[:200]}\nBear: {bear[:200]}\n"
        f"Answer ONLY with an integer between 1 and 10."
    )
    try:
        conv_str = safe_llm_call(conv_prompt)
        digits = re.sub(r'[^0-9]', '', conv_str)
        conviction = min(max(int(digits), 1), 10) if digits else 5
    except Exception:
        conviction = 5
    
    logger.info(f"[MULTI-ANALYZER] {symbol} → Conviction {conviction}/10")
    
    return {
        "symbol": symbol,
        "text_analysis": text_analysis,
        "bull_argument": bull,
        "bear_argument": bear,
        "conviction_score": conviction
    }


def multi_analyzer_node(state: AgentState):
    """(#13) Analyse tous les candidats du prospector en séquence et les classe."""
    candidates = state.get("candidates", [])
    if not candidates:
        return {"messages": ["Aucun candidat à analyser."], "final_decision": "HOLD"}
    
    logger.info(f"[MULTI-ANALYZER] Début de l'analyse de {len(candidates)} candidats...")
    
    analyzed = []
    msgs = []
    for c in candidates:
        sym = c.get("symbol", "?")
        result = _analyze_single_ticker(sym)
        analyzed.append(result)
        msgs.append(f"🔍 {sym} : Conviction {result['conviction_score']}/10")
    
    # Trier par conviction décroissante
    analyzed.sort(key=lambda x: x.get("conviction_score", 0), reverse=True)
    
    # Le gagnant
    winner = analyzed[0]
    ranking = " > ".join([f"{a['symbol']}({a['conviction_score']})" for a in analyzed])
    
    logger.info(f"[MULTI-ANALYZER] Ranking: {ranking}")
    msgs.append(f"\n🏆 FINAL RANKING: {ranking}")
    msgs.append(f"⭐ WINNER: {winner['symbol']} with a conviction of {winner['conviction_score']}/10")
    
    return {
        "symbol": winner["symbol"],
        "text_analysis": winner["text_analysis"],
        "bull_argument": winner["bull_argument"],
        "bear_argument": winner["bear_argument"],
        "candidates": analyzed,
        "messages": msgs
    }


def thinker_text_node(state: AgentState):
    """(#1) Fetches REAL news + (#4) fundamental data for the specific ticker."""
    symbol = state.get('symbol', 'UNKNOWN')
    logger.info(f"[TEXT NODE] Fetching real news and fundamentals for {symbol}...")
    
    parts = []
    
    try:
        import asyncio
        news = asyncio.run(news_pipeline_agent.analyze_market_sentiment_combined(symbol))
        parts.append(news)
    except Exception as e:
        parts.append(f"News search failed: {e}")
    
    fundamentals = _fetch_fundamentals(symbol)
    parts.append(fundamentals)
    
    text_result = "\n\n".join(parts)
    logger.info(f"[TEXT NODE] Complete analysis for {symbol} ({len(text_result)} characters)")
    return {"text_analysis": text_result, "messages": [f"Real analysis for {symbol} completed (news + fundamentals)."]}


def bull_node(state: AgentState):
    """The Optimistic analyst finds all the reasons to buy."""
    text = state.get("text_analysis", "")
    symbol = state.get("symbol", "")
    print(f"[BULL NODE] Searching for bullish arguments for {symbol}...")
    
    prompt = f"You are a multimillionaire trader with over 30 years of market experience. We are analyzing stock {symbol}.\nNews: '{text}'.\n\nLeverage your three decades of experience to find 2 to 3 powerful arguments justifying why this is an excellent BUY opportunity. Be confident but speak like a veteran who has seen it all."
    
    try:
        response = get_content(get_llm().invoke(prompt))
    except Exception as e:
        response = f"Bull Error: {str(e)}"
        
    return {"bull_argument": response, "messages": [f"🐂 Bull Thesis: {response}"]}


def bear_node(state: AgentState):
    """The Pessimistic analyst finds all the reasons to sell."""
    text = state.get("text_analysis", "")
    symbol = state.get("symbol", "")
    print(f"[BEAR NODE] Searching for bearish arguments for {symbol}...")
    
    prompt = f"You are a fearsome short-seller with over 30 years of ruthless market experience (you predicted the 2008 crash). We are analyzing stock {symbol}.\nNews: '{text}'.\n\nFind 2 to 3 cynical and deep arguments justifying why this stock is a tourist trap and that you should SELL or Short. Demolish false hopes with the coldness of a veteran."
    
    try:
        response = get_content(get_llm().invoke(prompt))
    except Exception as e:
        response = f"Bear Error: {str(e)}"
        
    return {"bear_argument": response, "messages": [f"🐻 Bear Thesis: {response}"]}


def judge_node(state: AgentState):
    """The Judge listens to the Bull and the Bear and decides impartially."""
    import os
    bull = state.get("bull_argument", "")
    bear = state.get("bear_argument", "")
    symbol = state.get("symbol", "")
    print(f"[JUDGE NODE] Final deliberation for {symbol}. Querying memory base (ChromaDB)...")
    
    try:
        db = get_vector_db()
        search_query = f"Symbol: {symbol}. Bullish: {bull[:100]}. Bearish: {bear[:100]}."
        results = db.similarity_search(search_query, k=3)
        macro_docs = db.similarity_search("macro outlook", k=1, filter={"source": "macro_outlook"})
        
        saved_strategy = ""
        if macro_docs:
            saved_strategy += f"\n\n{macro_docs[0].page_content}"
            
        if results:
            memories = "\n".join([f"- {r.page_content}" for r in results])
            saved_strategy += f"\n\nCONTEXTUAL MEMORIES AND PAST LESSONS (ChromaDB) :\n{memories}"
            
        if not saved_strategy:
            saved_strategy = "\n\nNo past context found."
    except Exception as e:
        print(f"ChromaDB access error: {e}")
        saved_strategy = ""
            
    prompt = f"You are the impartial FINAL JUDGE, a legendary fund manager with over 30 years of stock market experience. Your UNIQUE goal is to generate **20% annual return** for your investors. You have no room for emotion. The asset is {symbol}.\n\nBULL Arguments: '{bull}'\n\nBEAR Arguments: '{bear}'.\n{saved_strategy}\n\nConfront these two theses with clarity and the extreme prudence of your decades of experience. Ensure your decision is aligned with the 6-MONTH MACRO VISION. First, write 2 to 3 sentences to summarize your arbitrage from your perspective as an old Wall Street wolf.\nThen, on the very LAST line, give your final trading verdict with EXACTLY ONE WORD from: BUY, SELL, or HOLD."
    
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        content = get_content(response)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        decision = lines[-1].upper() if lines else "HOLD"
        # Extract BUY/SELL/HOLD even if the last line contains other words
        import re as re_judge
        for keyword in ["BUY", "SELL", "HOLD"]:
            if keyword in decision:
                decision = keyword
                break
        reasoning = " ".join(lines[:-1]) if len(lines) > 1 else "Arbitrage rendered."
        
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
    except Exception as e:
        decision = "HOLD"
        reasoning = f"Judge error during deliberation: {str(e)}"
        
    global GLOBAL_DEBATES_CACHE
    GLOBAL_DEBATES_CACHE.append({
        "symbol": symbol,
        "timestamp": time.time(),
        "bull": bull,
        "bear": bear,
        "judge": reasoning,
        "decision": decision
    })
    if len(GLOBAL_DEBATES_CACHE) > 10:
        GLOBAL_DEBATES_CACHE.pop(0)
        
    return {"final_decision": decision, "messages": [f"⚖️ Judge's Verdict: {reasoning}", f"The AI agent decided: {decision} for {symbol}"]}


def risk_manager_node(state: AgentState):
    """
    Modular Risk Manager v2.
    Allocation (% of engaged capital) is calculated dynamically via 2 LLM scores:
      - Market DANGER score (1-10): evaluates volatility/macro risk
      - Trade CONVICTION score (1-10): evaluates the quality of the opportunity
    
    Formula: allocation% = base_max * (conviction / 10) * (1 - (danger - 1) / 18)
      → Max conviction (10) + Min danger (1)  = 20% of capital (ceiling)
      → Min conviction (1)  + Max danger (10) = ~1% of capital (floor)
    """
    import os, math, re
    decision = state.get("final_decision", "HOLD")
    symbol = state.get("symbol", "")
    bull = state.get("bull_argument", "")
    bear = state.get("bear_argument", "")
    print(f"[RISK MANAGER NODE] Calculating modular sizing for {decision} on {symbol}")
    
    if decision not in ["BUY", "SELL"]:
        return {"trade_notional": 0.0, "trade_qty": 0.0, "messages": ["Risk Management: Holding state (HOLD), no financial transaction planned."]}
        
    try:
        from alpaca.trading.client import TradingClient
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or api_key == "your_alpaca_api_key":
            return {"messages": ["Risk Management Error: Alpaca keys not configured."]}
            
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        equity = float(account.equity)
        
        trade_notional = 0.0
        trade_qty = 0.0
        tp_price = 0.0
        sl_price = 0.0
        trail_percent = 0.0
        danger_score = 5
        conviction_score = 5

        if decision == "BUY":
            # --- (#2) Check existing position ---
            try:
                existing_pos = client.get_open_position(symbol)
                if existing_pos:
                    msg = f"⚠️ Position already open for {symbol} ({existing_pos.qty} shares, P&L: {existing_pos.unrealized_pl}$). Buy blocked."
                    logger.info(f"[RISK MANAGER] {msg}")
                    return {"final_decision": "HOLD", "trade_notional": 0.0, "trade_qty": 0.0, "messages": [f"🛡️ Risk Manager: {msg}"]}
            except Exception:
                pass  # No existing position = OK to buy
            
            # --- (#3) Check diversification ---
            try:
                all_positions = client.get_all_positions()
                if len(all_positions) >= 25:
                    msg = f"⚠️ Already {len(all_positions)} positions open. Diversification: buy blocked (max 25)."
                    logger.info(f"[RISK MANAGER] {msg}")
                    return {"final_decision": "HOLD", "trade_notional": 0.0, "trade_qty": 0.0, "messages": [f"🛡️ Risk Manager: {msg}"]}
            except Exception:
                pass
                
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest
            latest_price = 0.0
            try:
                data_client = StockHistoricalDataClient(api_key, secret_key)
                req = StockLatestTradeRequest(symbol_or_symbols=symbol)
                latest_trade = data_client.get_stock_latest_trade(req)
                latest_price = latest_trade[symbol].price
            except Exception as e:
                print(f"Impossible to fetch live price: {e}")
                
            if latest_price > 0:
                # --- DOUBLE LLM EVALUATION ---
                llm = get_llm()
                
                # 1. Macro DANGER score (market risk)
                prompt_danger = (
                    f"You are an institutional risk manager with 30 years of experience. "
                    f"Rate the current market DANGER level from 1 to 10 for buying {symbol} "
                    f"(volatility, macro risk, geopolitical context). "
                    f"Answer ONLY with an integer between 1 and 10."
                )
                try:
                    score_str = get_content(llm.invoke(prompt_danger))
                    digits = re.sub(r'[^0-9]', '', score_str)
                    danger_score = min(max(int(digits), 1), 10) if digits else 5
                except Exception:
                    danger_score = 5

                # 2. Trade CONVICTION score (quality of opportunity)
                prompt_conviction = (
                    f"You are a legendary fund manager. Reading the following arguments about {symbol}, "
                    f"give a CONVICTION score from 1 to 10 on the intrinsic quality of this trade "
                    f"(1 = very weak/random signal, 10 = exceptional and asymmetric opportunity).\n"
                    f"BULL Argument: {bull[:300]}\n"
                    f"BEAR Argument: {bear[:300]}\n"
                    f"Answer ONLY with an integer between 1 and 10."
                )
                try:
                    conv_str = get_content(llm.invoke(prompt_conviction))
                    digits = re.sub(r'[^0-9]', '', conv_str)
                    conviction_score = min(max(int(digits), 1), 10) if digits else 5
                except Exception:
                    conviction_score = 5

                # --- MODULAR ALLOCATION CALCULATION ---
                # Base max = 20%. Adjust according to conviction (↑) and danger (↓)
                base_max = 0.10
                conviction_factor = conviction_score / 10.0            # 0.1 → 1.0
                danger_penalty    = 1.0 - (danger_score - 1) / 18.0   # 0.5 → 1.0
                allocation_pct = base_max * conviction_factor * danger_penalty
                allocation_pct = min(max(allocation_pct, 0.01), 0.10) # clamp [1%, 10%]
                allowed_allocation = equity * allocation_pct
                
                trade_qty = math.floor(allowed_allocation / latest_price)
                
                if trade_qty > 0:
                    # Dynamic Trailing Stop-Loss based on ATR
                    text = state.get("text_analysis", "")
                    atr_val = latest_price * 0.02
                    match = re.search(r'Volatility \(ATR\): ([\d\.]+)\$', text)
                    if match:
                        atr_val = float(match.group(1))
                    
                    # Trailing Stop distance definition (trail_percent)
                    # High danger = tight stop (1.5x ATR). Very low danger = wide stop (3.0x ATR)
                    multiplier = 1.5 if danger_score >= 8 else (3.0 if danger_score <= 3 else 2.0)
                    trail_dist = atr_val * multiplier
                    trail_percent = (trail_dist / latest_price) * 100
                    trail_percent = min(max(trail_percent, 1.0), 10.0) # clamp [1%, 10%]
                    
                    msg = (
                        f"📊 MODULAR Allocation: {allocation_pct*100:.1f}% of capital ({allowed_allocation:.0f}$) → "
                        f"{trade_qty} shares {symbol} at {latest_price}$ | "
                        f"Danger: {danger_score}/10 · Conviction: {conviction_score}/10 | "
                        f"Trailing Stop enabled: -{trail_percent:.1f}% (ATR: {atr_val:.2f}$ x{multiplier})"
                    )
                else:
                    decision = "HOLD"
                    msg = f"Risk Block: Asset {symbol} ({latest_price}$) too expensive for the calculated allocation of {allowed_allocation:.2f}$."
            else:
                decision = "HOLD"
                msg = "Could not fetch price. Trade cancelled for security."
            
        elif decision == "SELL":
            try:
                position = client.get_open_position(symbol)
                trade_qty = float(position.qty)
                msg = f"Exact Liquidation: Order to sell the entire position of {trade_qty} shares on {symbol}."
            except Exception as e:
                trade_qty = 1.0
                msg = f"No position found for {symbol}. Experimental short (1 share)."
        else:
            msg = "Alpaca Error: Could not verify risk."

        return {
            "final_decision": decision, 
            "trade_notional": trade_notional, 
            "trade_qty": trade_qty, 
            "tp_price": tp_price,
            "sl_price": sl_price,
            "trail_percent": trail_percent if 'trail_percent' in locals() else 0.0,
            "danger_score": danger_score,
            "messages": [f"🛡️ Risk Manager: {msg}"]
        }
    except Exception as e:
         logger.error(f"[RISK MANAGER] Error: {e}")
         return {"messages": [f"Risk Node Error: {str(e)}"]}


def actor_node(state: AgentState):
    """Executes the trade on Alpaca Paper Trading based on Risk Manager allocation."""
    import os
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    
    decision = state.get("final_decision", "HOLD")
    symbol = state.get("symbol", "")
    trade_notional = state.get("trade_notional", 0.0)
    trade_qty = state.get("trade_qty", 0.0)
    tp_price = state.get("tp_price", 0.0)
    sl_price = state.get("sl_price", 0.0)
    
    logger.info(f"[ACTOR NODE] Executing decision: {decision} for {symbol}")
    
    msg = f"No action taken for {symbol}. Decision: {decision}."
    
    if decision in ["BUY", "SELL"]:
        if trade_notional == 0 and trade_qty == 0:
             return {"messages": ["Trade cancelled because the quantity evaluated by the risk manager is zero."]}
             
        try:
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            
            if not api_key or api_key == "your_alpaca_api_key":
                msg = "Trade cancelled: Missing Alpaca keys."
            else:
                client = TradingClient(api_key, secret_key, paper=True)
                side = OrderSide.BUY if decision == "BUY" else OrderSide.SELL
                
                trade_executed = False
                
                if trade_notional > 0:
                     # Alpaca allows fractional shares using Notional constraint. Only Market DAY is allowed for notional
                     market_order_data = MarketOrderRequest(
                         symbol=symbol,
                         notional=round(trade_notional, 2),
                         side=side,
                         time_in_force=TimeInForce.DAY
                     )
                     order = client.submit_order(order_data=market_order_data)
                     msg = f"Success: {decision} order of {trade_notional:.2f}$ ({symbol}) executed (Fractional). (ID: {order.id})"
                     trade_executed = True
                else:
                     if trade_qty == 0: trade_qty = 1.0
                     
                     trail_percent = state.get("trail_percent", 0.0)
                     
                     market_order_data = MarketOrderRequest(
                         symbol=symbol,
                         qty=int(trade_qty),
                         side=side,
                         time_in_force=TimeInForce.DAY
                     )
                     order = client.submit_order(order_data=market_order_data)
                     msg_suffix = " (Standard)"
                     
                     if side == OrderSide.BUY and trail_percent > 0:
                         from alpaca.trading.requests import TrailingStopOrderRequest
                         time.sleep(1) # Wait for Market execution
                         ts_req = TrailingStopOrderRequest(
                             symbol=symbol,
                             qty=int(trade_qty),
                             side=OrderSide.SELL,
                             time_in_force=TimeInForce.GTC,
                             trail_percent=round(trail_percent, 2)
                         )
                         ts_order = client.submit_order(order_data=ts_req)
                         msg_suffix = f" with Secured Trailing Stop (-{trail_percent:.1f}%)"

                     msg = f"Success: {decision} order of {trade_qty} shares {symbol} executed{msg_suffix}. (ID: {order.id})"
                     trade_executed = True
                     
                # Vectorial insertion of this action
                if trade_executed:
                    try:
                        import time
                        from langchain_core.documents import Document
                        
                        judge_reasoning = next((m for m in state.get("messages", []) if "Judge's Verdict" in m), "Unknown reasoning")
                        memory_context = f"I executed an experimental {decision} order on stock {symbol}. Reason: {judge_reasoning}."
                        
                        db = get_vector_db()
                        doc = Document(page_content=memory_context, metadata={"source": "trade", "symbol": symbol, "timestamp": time.time()})
                        db.add_documents([doc])
                        logger.info(f"[ACTOR NODE] Archived in deep memory ({symbol} -> {decision}).")
                    except Exception as ex:
                        logger.warning(f"[ACTOR NODE] ChromaDB memorization failed: {ex}")
                     
        except Exception as e:
            msg = f"Alpaca Exchange error during {decision} trade for {symbol}: {str(e)}"
            
    return {"messages": [msg]}


def router(state: AgentState):
    """Determines if we need to prospect a new stock or immediately analyze an existing one."""
    if not state.get("symbol"):
        return "prospector"
    return "text_thinker"


# Define the LangGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("prospector", prospector_node)
workflow.add_node("multi_analyzer", multi_analyzer_node)
workflow.add_node("text_thinker", thinker_text_node)
workflow.add_node("bull_node", bull_node)
workflow.add_node("bear_node", bear_node)
workflow.add_node("judge_node", judge_node)
workflow.add_node("risk_manager", risk_manager_node)
workflow.add_node("actor", actor_node)

# Define Data Flow with Conditional Entry
workflow.set_conditional_entry_point(
    router,
    {"prospector": "prospector", "text_thinker": "text_thinker"}
)

# (#13) Prospector → Multi-Analyzer (5 tickers, analysis + ranking) → Judge → Risk Manager → Actor
workflow.add_edge("prospector", "multi_analyzer")
workflow.add_edge("multi_analyzer", "judge_node")

# Direct entry with a specific symbol (manual mode)
workflow.add_edge("text_thinker", "bull_node")
workflow.add_edge("bull_node", "bear_node")
workflow.add_edge("bear_node", "judge_node")

workflow.add_edge("judge_node", "risk_manager")
workflow.add_edge("risk_manager", "actor")
workflow.add_edge("actor", END)

# Compile graph into runnable agent
trading_agent = workflow.compile()

def chat_with_agent(user_message: str, portfolio_context: str = "") -> str:
    """Queries the LLM agent with local ChromaDB context for a conversational interface."""
    from langchain_core.messages import SystemMessage, HumanMessage
    
    try:
        db = get_vector_db()
        results = db.similarity_search(user_message, k=5)
        memories = "\n".join([f"- {r.page_content}" for r in results])
    except Exception as e:
        memories = f"Error accessing my deep ChromaDB memory: {e}"
        
    system_prompt = f"""You are the Artificial Intelligence of this Autonomous Trading system, but you have the persona of a ruthless veteran trader with over 30 years of experience (you survived 1987, 2000, and 2008). Your performance goal is targeted at 20% annually.
The user is asking you a direct question about your current or past stock choices.
Here is a summary of the current state of their Alpaca portfolio:
{portfolio_context}

Here are your recent memories retrieved from the ChromaDB vector base related to the question:
{memories}

Strict guidelines:
1. Be direct, realistic, and a bit cynical (Wall Street wolf style).
2. Justify your decisions based on your ChromaDB memories with the wisdom of your 30 years of experience.
3. No useless chatter. Speak about yourself in the first person.
4. You MUST answer in English."""
    
    llm = get_llm()
    try:
        msg = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        response = get_content(llm.invoke(msg))
        return response
    except Exception as e:
        return f"Error communicating with my LLM brain: {str(e)}"

if __name__ == "__main__":
    print("Testing Prospector (No Symbol)...")
    result = trading_agent.invoke({"symbol": "", "messages": []})
    print(f"Decision: {result.get('final_decision')}, Symbol: {result.get('symbol')}")
