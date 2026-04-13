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

# --- Optimisation du Logging Tiers ---
logging.getLogger("ddgs").setLevel(logging.ERROR)
logging.getLogger("primp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

import json

GLOBAL_DEBATES_CACHE = []

def safe_ddg_search(query: str, max_retries: int = 3) -> str:
    """Wrapper résilient pour DuckDuckGo. Retry avec backoff en cas d'erreur réseau."""
    search = DuckDuckGoSearchResults()
    for attempt in range(max_retries):
        try:
            result = search.invoke(query)
            if result:
                # Retourne simplement le résultat textuel brut pour la LLM (sans FinBERT)
                return result
        except Exception as e:
            logger.warning(f"[DDG] Tentative {attempt+1}/{max_retries} échouée pour '{query[:50]}...' : {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # backoff : 1s, 2s, 4s
    return f"Recherche indisponible pour '{query}' après {max_retries} tentatives."

# --- Logging structuré (#8) ---
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
        logger.warning("GOOGLE_API_KEY manquante ou placeholder. Les requêtes échoueront.")
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
        
        pos_str = "\n".join([f"- {p.symbol}: {p.qty} actions à ${p.current_price} (P&L latent: {p.unrealized_pl}$)" for p in positions])
        if not pos_str:
            pos_str = "Aucune position ouverte."
            
        context = f"Solde total: ${account.equity}\nCash disponible: ${account.cash}\nPositions ouvertes:\n{pos_str}"
        prompt = f"Tu es un expert financier. Voici l'état actuel de mon portefeuille de trading autonome :\n{context}\n\nRédige une courte analyse (3-4 phrases en français) de ces résultats. Félicite ou mets en garde en fonction du P&L, et donne un conseil rapide sur l'exposition."
        
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
            status = "Succès" if pl_pct > 0 else "Echec"
            track_doc = (
                f"Trade {p.symbol} : Entré à {p.avg_entry_price}$, "
                f"actuellement {p.current_price}$ ({pl_pct:+.1f}%). {status}. "
                f"Position de {p.qty} actions, P&L latent: {p.unrealized_pl}$."
            )
            doc = Document(page_content=track_doc, metadata={"source": "performance_track", "symbol": p.symbol, "timestamp": time.time()})
            db.add_documents([doc])
        
        logger.info(f"[REFLECTION] {len(positions)} positions trackées dans ChromaDB.")
        
        pos_str = "\n".join([f"- {p.symbol}: {p.qty} actions, P&L Latent: {p.unrealized_pl}$ ({float(p.unrealized_plpc)*100 if p.unrealized_plpc else 0:.1f}%)" for p in positions])
        if not pos_str:
            pos_str = "Aucune position. Historique vierge ou tout a été clôturé."
            
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(period="1M", timeframe="1D")
        history = client.get_portfolio_history(req)
        profit_loss_1m = history.equity[-1] - history.equity[0] if history.equity else 0
        
        context = f"Portefeuille actuel: {account.equity}$. P&L sur 1 mois: {profit_loss_1m}$.\nPositions ouvertes:\n{pos_str}"
        prompt = f"Tu es l'esprit critique d'un algorithme de trading. Voici l'état du portefeuille :\n{context}\n\nIdentifie les erreurs (ex: positions très perdantes) et les succès. Sur la base de ces gains/pertes, rédige 3 règles strictes et concises (en français) que tu devras suivre pour ton prochain trade (ex: éviter tel secteur, couper rapidement les pertes). Ne renvoie QUE les 3 règles sous forme de tirets."
        
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
            f"Tu es le stratège en chef d'un fond d'investissement de très haut niveau.\n"
            f"Voici les dernières actualités économiques mondiales :\n{news}\n\n"
            f"Rédige une vision stratégique concise et tranchante (max 4-5 phrases) qui :\n"
            f"1. Résume le sentiment macro-économique actuel.\n"
            f"2. Identifie les défis à venir et les secteurs les plus résilients/prometteurs pour les 6 prochains mois (ainsi que ceux à fuir absolument).\n"
            f"Ne donne que ton analyse, sans formule de politesse."
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
                logger.warning(f"Impossible de récupérer les positions Alpaca pour la prospection : {e}")

        # Recherche plus globale, moins centrée "tech"
        news = safe_ddg_search("top stocks to buy today promising unvalued different sectors healthcare energy finance tech consumer")
        
        # Règles stratégiques depuis ChromaDB
        saved_strategy = ""
        try:
            db = get_vector_db()
            macro_docs = db.similarity_search("macro outlook", k=1, filter={"source": "macro_outlook"})
            if macro_docs:
                saved_strategy = f"\n\n{macro_docs[0].page_content}"
                
            strategy_docs = db.similarity_search("règles stratégiques trading erreurs succès", k=3)
            if strategy_docs:
                rules = "\n".join([f"- {d.page_content}" for d in strategy_docs])
                saved_strategy += f"\n\nRÈGLES STRATÉGIQUES RÉCENTES :\n{rules}"
        except Exception as e:
            logger.warning(f"ChromaDB indisponible pour la stratégie : {e}")

        prompt = (
            f"Tu es un gestionnaire de fonds impitoyable avec plus de 30 ans d'expérience. Ton objectif : 20% annuel tout en GARANTISSANT UNE DIVERSIFICATION SECTORIELLE STRICTE.\n\n"
            f"Voici mes positions actuelles : {current_positions_str}\n"
            f"Analyse implicitement les secteurs de mes avoirs actuels (ex: Tech, Finance, Santé).\n"
            f"En te basant sur les actualités et SUR LA VISION MACRO À 6 MOIS (ci-dessous), choisis EXACTEMENT 5 codes boursiers (tickers) qui présentent les meilleures opportunités.\n"
            f"CRITÈRE MAJEUR 1 : Diversification. Tu DOIS sélectionner des actions issues de catégories DIVERSES (ex: si j'ai beaucoup d'IA/Tech, choisis de la Santé, de l'Énergie, etc.).\n"
            f"CRITÈRE MAJEUR 2 : Tendance long-terme. Préfère les sociétés adaptées aux perspectives économiques mondiales (taux, inflation).\n\n"
            f"Actualités du jour : {news}\n"
            f"{saved_strategy}\n\n"
            f"Pour chaque ticker, écris une ligne au format : TICKER | Secteur | Justification courte (1 phrase)\n"
            f"Exemple :\nJNJ | Santé | Dividendes solides et protection contre la volatilité tech.\nXOM | Énergie | Hausse du baril et fondamentaux très sous-évalués.\n\n"
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
            "messages": [f"Erreur lors de la recherche Web: {str(e)}"]
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
            
            # Formattage et alertes
            signal = "Golden Cross (↑)" if sma_20 > sma_50 else "Death Cross (↓)"
            trend = "↑ HAUSSIER" if current_price > sma_50 else "↓ BAISSIER"
            
            rsi_alert = "SURACHETÉ (RSI>70)" if rsi_14 > 70 else ("SURVENDU (RSI<30)" if rsi_14 < 30 else "Neutre")
            macd_signal = "HAUSSIER (MACD>0)" if macd_val > 0 else "BAISSIER (MACD<0)"
            
            return (
                f"\n📊 DONNÉES TECHNIQUES [{symbol}] | Prix Actuel: {current_price:.2f}$ \n"
                f"- Tendance: {trend} (SMA20: {sma_20:.2f}$, SMA50: {sma_50:.2f}$ → {signal})\n"
                f"- Volatilité (ATR): {atr_14:.2f}$ par jour (Mouvement moyen attendu).\n"
                f"- Momentum: RSI={rsi_14:.1f} ({rsi_alert}) | MACD: {macd_signal}"
            )
        return f"Barres historiques indisponibles pour {symbol}."
    except Exception as e:
        return f"Fondamentaux indisponibles pour {symbol}: {e}"


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
        f"Tu es un trader multi-millionnaire avec 30 ans d'expérience. Action {symbol}.\n"
        f"Données: '{text_analysis[:500]}'.\n"
        f"Trouve 2 arguments puissants justifiant un ACHAT. Maximum 3 phrases."
    )
    try:
        bull = safe_llm_call(bull_prompt)
    except Exception as e:
        bull = f"Erreur Bull : {e}"
    
    # 4. Bear
    bear_prompt = (
        f"Tu es un short-seller redoutable avec 30 ans d'expérience. Action {symbol}.\n"
        f"Données: '{text_analysis[:500]}'.\n"
        f"Trouve 2 arguments cyniques justifiant une VENTE. Maximum 3 phrases."
    )
    try:
        bear = safe_llm_call(bear_prompt)
    except Exception as e:
        bear = f"Erreur Bear : {e}"
    
    # 5. Score de conviction
    conv_prompt = (
        f"Score de CONVICTION de 1 à 10 pour un trade sur {symbol}.\n"
        f"Bull: {bull[:200]}\nBear: {bear[:200]}\n"
        f"Réponds UNIQUEMENT par un entier entre 1 et 10."
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
    
    logger.info(f"[MULTI-ANALYZER] Classement : {ranking}")
    msgs.append(f"\n🏆 CLASSEMENT FINAL : {ranking}")
    msgs.append(f"⭐ GAGNANT : {winner['symbol']} avec une conviction de {winner['conviction_score']}/10")
    
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
        parts.append(f"Recherche d'actualités échouée : {e}")
    
    fundamentals = _fetch_fundamentals(symbol)
    parts.append(fundamentals)
    
    text_result = "\n\n".join(parts)
    logger.info(f"[TEXT NODE] Analyse complète pour {symbol} ({len(text_result)} caractères)")
    return {"text_analysis": text_result, "messages": [f"Analyse réelle pour {symbol} terminée (news + fondamentaux)."]}


def bull_node(state: AgentState):
    """L'analyste Optimiste trouve toutes les raisons d'acheter."""
    text = state.get("text_analysis", "")
    symbol = state.get("symbol", "")
    print(f"[BULL NODE] Recherche d'arguments haussiers pour {symbol}...")
    
    prompt = f"Tu es un trader multi-millionnaire avec plus de 30 ans d'expérience de marché. On analyse l'action {symbol}.\nActualités: '{text}'.\n\nTire parti de tes trois décennies d'expérience pour trouver 2 à 3 arguments puissants justifiant pourquoi c'est une excellente opportunité d'ACHAT. Sois confiant mais parle comme un vétéran qui a tout vu."
    
    try:
        response = get_content(get_llm().invoke(prompt))
    except Exception as e:
        response = f"Erreur du Bull : {str(e)}"
        
    return {"bull_argument": response, "messages": [f"🐂 Thèse Haussière (Bull) : {response}"]}


def bear_node(state: AgentState):
    """L'analyste Pessimiste trouve toutes les raisons de vendre."""
    text = state.get("text_analysis", "")
    symbol = state.get("symbol", "")
    print(f"[BEAR NODE] Recherche d'arguments baissiers pour {symbol}...")
    
    prompt = f"Tu es un short-seller redoutable avec plus de 30 ans d'expérience impitoyable de marché (tu as anticipé le krach de 2008). On analyse l'action {symbol}.\nActualités: '{text}'.\n\nTrouve 2 à 3 arguments cyniques et profonds justifiant pourquoi cette action est un piège à touristes et qu'il faut VENDRE ou Shorter. Démolis les faux espoirs avec la froideur d'un vétéran."
    
    try:
        response = get_content(get_llm().invoke(prompt))
    except Exception as e:
        response = f"Erreur du Bear : {str(e)}"
        
    return {"bear_argument": response, "messages": [f"🐻 Thèse Baissière (Bear) : {response}"]}


def judge_node(state: AgentState):
    """Le Juge écoute le Bull et le Bear et tranche de manière impartiale."""
    import os
    bull = state.get("bull_argument", "")
    bear = state.get("bear_argument", "")
    symbol = state.get("symbol", "")
    print(f"[JUDGE NODE] Délibération finale pour {symbol}. Interrogation de la base de souvenirs (ChromaDB)...")
    
    try:
        db = get_vector_db()
        search_query = f"Symbol: {symbol}. Optimiste: {bull[:100]}. Pessimiste: {bear[:100]}."
        results = db.similarity_search(search_query, k=3)
        macro_docs = db.similarity_search("macro outlook", k=1, filter={"source": "macro_outlook"})
        
        saved_strategy = ""
        if macro_docs:
            saved_strategy += f"\n\n{macro_docs[0].page_content}"
            
        if results:
            memories = "\n".join([f"- {r.page_content}" for r in results])
            saved_strategy += f"\n\nSOUVENIRS CONTEXTUELS ET LEÇONS PASSÉES (ChromaDB) :\n{memories}"
            
        if not saved_strategy:
            saved_strategy = "\n\nAucun contexte passé trouvé."
    except Exception as e:
        print(f"Erreur d'accès ChromaDB: {e}")
        saved_strategy = ""
            
    prompt = f"Tu es le JUGE FINAL impartial, un gérant de fonds légendaire avec plus de 30 ans d'expérience boursière. Ton UNIQUE objectif est de générer **20% de rendement annuel** pour tes investisseurs. Tu n'as pas de place pour l'émotion. L'actif est {symbol}.\n\nArguments du BULL : '{bull}'\n\nArguments du BEAR : '{bear}'.\n{saved_strategy}\n\nConfronte ces deux thèses avec lucidité et l'extrême prudence de tes décennies d'expérience. Assure-toi que ta décision est alignée avec la VISION MACRO À 6 MOIS. D'abord, écris 2 à 3 phrases pour résumer ton arbitrage avec ton regard de vieux loup de Wall Street.\nEnsuite, sur la toute DERNIÈRE ligne, donne ton verdict final de trading avec EXACTEMENT UN MOT parmi : BUY, SELL, ou HOLD."
    
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        content = get_content(response)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        decision = lines[-1].upper() if lines else "HOLD"
        # Extraire BUY/SELL/HOLD même si la dernière ligne contient d'autres mots
        import re as re_judge
        for keyword in ["BUY", "SELL", "HOLD"]:
            if keyword in decision:
                decision = keyword
                break
        reasoning = " ".join(lines[:-1]) if len(lines) > 1 else "Arbitrage rendu."
        
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
    except Exception as e:
        decision = "HOLD"
        reasoning = f"Erreur du Juge lors de la délibération : {str(e)}"
        
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
        
    return {"final_decision": decision, "messages": [f"⚖️ Verdict du Juge : {reasoning}", f"L'agent IA a décidé : {decision} pour {symbol}"]}


def risk_manager_node(state: AgentState):
    """
    Risk Manager Modulaire v2.
    L'allocation (% du capital engagé) est calculée dynamiquement via 2 scores LLM :
      - Score de DANGER du marché  (1-10) : évalue la volatilité/risque macro
      - Score de CONVICTION du trade (1-10) : évalue la qualité de l'opportunité
    
    Formule : allocation% = base_max * (conviction / 10) * (1 - (danger - 1) / 18)
      → Conviction max (10) + Danger min (1)  = 20% du capital (plafond)
      → Conviction min (1)  + Danger max (10) = ~1% du capital (plancher)
    """
    import os, math, re
    decision = state.get("final_decision", "HOLD")
    symbol = state.get("symbol", "")
    bull = state.get("bull_argument", "")
    bear = state.get("bear_argument", "")
    print(f"[RISK MANAGER NODE] Calcul du dimensionnement modulaire pour {decision} sur {symbol}")
    
    if decision not in ["BUY", "SELL"]:
        return {"trade_notional": 0.0, "trade_qty": 0.0, "messages": ["Gestion des risques : Maintien de l'état (HOLD), aucune transaction financière prévue."]}
        
    try:
        from alpaca.trading.client import TradingClient
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or api_key == "votre_cle_api_alpaca":
            return {"messages": ["Erreur Gestion Risque : Clés Alpaca non configurées."]}
            
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
            # --- (#2) Vérification position existante ---
            try:
                existing_pos = client.get_open_position(symbol)
                if existing_pos:
                    msg = f"⚠️ Position déjà ouverte sur {symbol} ({existing_pos.qty} parts, P&L: {existing_pos.unrealized_pl}$). Achat bloqué."
                    logger.info(f"[RISK MANAGER] {msg}")
                    return {"final_decision": "HOLD", "trade_notional": 0.0, "trade_qty": 0.0, "messages": [f"🛡️ Gestionnaire des Risques : {msg}"]}
            except Exception:
                pass  # Pas de position existante = OK pour acheter
            
            # --- (#3) Vérification diversification ---
            try:
                all_positions = client.get_all_positions()
                if len(all_positions) >= 25:
                    msg = f"⚠️ Déjà {len(all_positions)} positions ouvertes. Diversification : achat bloqué (max 25)."
                    logger.info(f"[RISK MANAGER] {msg}")
                    return {"final_decision": "HOLD", "trade_notional": 0.0, "trade_qty": 0.0, "messages": [f"🛡️ Gestionnaire des Risques : {msg}"]}
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
                print(f"Impossible de récupérer le prix en direct: {e}")
                
            if latest_price > 0:
                # --- DOUBLE ÉVALUATION LLM ---
                llm = get_llm()
                
                # 1. Score de DANGER macro (risque marché)
                prompt_danger = (
                    f"Tu es un risk manager institutionnel avec 30 ans d'expérience. "
                    f"Évalue de 1 à 10 le niveau de DANGER actuel du marché pour un achat sur {symbol} "
                    f"(volatilité, risque macro, contexte géopolitique). "
                    f"Réponds UNIQUEMENT par un entier entre 1 et 10."
                )
                try:
                    score_str = get_content(llm.invoke(prompt_danger))
                    digits = re.sub(r'[^0-9]', '', score_str)
                    danger_score = min(max(int(digits), 1), 10) if digits else 5
                except Exception:
                    danger_score = 5

                # 2. Score de CONVICTION du trade (qualité de l'opportunité)
                prompt_conviction = (
                    f"Tu es un gérant de fonds légendaire. En lisant les arguments suivants sur {symbol}, "
                    f"donne un score de CONVICTION de 1 à 10 sur la qualité intrinsèque de ce trade "
                    f"(1 = signal très faible/aléatoire, 10 = opportunité exceptionnelle et asymétrique).\n"
                    f"Argument BULL : {bull[:300]}\n"
                    f"Argument BEAR : {bear[:300]}\n"
                    f"Réponds UNIQUEMENT par un entier entre 1 et 10."
                )
                try:
                    conv_str = get_content(llm.invoke(prompt_conviction))
                    digits = re.sub(r'[^0-9]', '', conv_str)
                    conviction_score = min(max(int(digits), 1), 10) if digits else 5
                except Exception:
                    conviction_score = 5

                # --- CALCUL MODULAIRE DE L'ALLOCATION ---
                # Base max = 20%. On ajuste selon conviction (↑) et danger (↓)
                base_max = 0.10
                conviction_factor = conviction_score / 10.0            # 0.1 → 1.0
                danger_penalty    = 1.0 - (danger_score - 1) / 18.0   # 0.5 → 1.0
                allocation_pct = base_max * conviction_factor * danger_penalty
                allocation_pct = min(max(allocation_pct, 0.01), 0.10) # clamp [1%, 10%]
                allowed_allocation = equity * allocation_pct
                
                trade_qty = math.floor(allowed_allocation / latest_price)
                
                if trade_qty > 0:
                    # Trailing Stop-Loss dynamique basé sur l'ATR
                    text = state.get("text_analysis", "")
                    atr_val = latest_price * 0.02
                    match = re.search(r'Volatilité \(ATR\): ([\d\.]+)\$', text)
                    if match:
                        atr_val = float(match.group(1))
                    
                    # Définition de la distance du Stop-Suiveur (trail_percent)
                    # Danger fort = stop serré (1.5x ATR). Danger très faible = stop large (3.0x ATR)
                    multiplier = 1.5 if danger_score >= 8 else (3.0 if danger_score <= 3 else 2.0)
                    trail_dist = atr_val * multiplier
                    trail_percent = (trail_dist / latest_price) * 100
                    trail_percent = min(max(trail_percent, 1.0), 10.0) # clamp [1%, 10%]
                    
                    msg = (
                        f"📊 Allocation MODULAIRE : {allocation_pct*100:.1f}% du capital ({allowed_allocation:.0f}$) → "
                        f"{trade_qty} actions {symbol} à {latest_price}$ | "
                        f"Danger: {danger_score}/10 · Conviction: {conviction_score}/10 | "
                        f"Trailing Stop activé: -{trail_percent:.1f}% (ATR: {atr_val:.2f}$ x{multiplier})"
                    )
                else:
                    decision = "HOLD"
                    msg = f"Blocage Risque : Action {symbol} ({latest_price}$) trop chère pour l'allocation calculée de {allowed_allocation:.2f}$."
            else:
                decision = "HOLD"
                msg = "Impossible de récupérer le prix. Trade annulé par sécurité."
            
        elif decision == "SELL":
            try:
                position = client.get_open_position(symbol)
                trade_qty = float(position.qty)
                msg = f"Liquidation exacte : Ordonnance de vendre l'intégralité de la position de {trade_qty} parts sur {symbol}."
            except Exception as e:
                trade_qty = 1.0
                msg = f"Aucune position trouvée sur {symbol}. Short expérimental (1 part)."
        else:
            msg = "Erreur Alpaca : Impossible de vérifier le risque."

        return {
            "final_decision": decision, 
            "trade_notional": trade_notional, 
            "trade_qty": trade_qty, 
            "tp_price": tp_price,
            "sl_price": sl_price,
            "trail_percent": trail_percent if 'trail_percent' in locals() else 0.0,
            "danger_score": danger_score,
            "messages": [f"🛡️ Gestionnaire des Risques : {msg}"]
        }
    except Exception as e:
         logger.error(f"[RISK MANAGER] Erreur : {e}")
         return {"messages": [f"Erreur du Nœud de Risque : {str(e)}"]}


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
    
    msg = f"Aucune action menée pour {symbol}. Décision: {decision}."
    
    if decision in ["BUY", "SELL"]:
        if trade_notional == 0 and trade_qty == 0:
             return {"messages": ["Trade annulé car la quantité évaluée par le gestionnaire de risques est à zéro."]}
             
        try:
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            
            if not api_key or api_key == "votre_cle_api_alpaca":
                msg = "Trade annulé : Clés Alpaca manquantes."
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
                     msg = f"Succès : Ordre {decision} de {trade_notional:.2f}$ ({symbol}) exécuté (Fractionnaire). (ID: {order.id})"
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
                         time.sleep(1) # Attente de l'exécution Market
                         ts_req = TrailingStopOrderRequest(
                             symbol=symbol,
                             qty=int(trade_qty),
                             side=OrderSide.SELL,
                             time_in_force=TimeInForce.GTC,
                             trail_percent=round(trail_percent, 2)
                         )
                         ts_order = client.submit_order(order_data=ts_req)
                         msg_suffix = f" avec Trailing Stop Sécurisé (-{trail_percent:.1f}%)"

                     msg = f"Succès : Ordre {decision} de {trade_qty} actions {symbol} exécuté{msg_suffix}. (ID: {order.id})"
                     trade_executed = True
                     
                # Insertion Vectorielle de ce passage à l'acte
                if trade_executed:
                    try:
                        import time
                        from langchain_core.documents import Document
                        
                        judge_reasoning = next((m for m in state.get("messages", []) if "Verdict du Juge" in m), "Raisonnement inconnu")
                        memory_context = f"J'ai passé un ordre expérimental {decision} sur l'action {symbol}. Raison : {judge_reasoning}."
                        
                        db = get_vector_db()
                        doc = Document(page_content=memory_context, metadata={"source": "trade", "symbol": symbol, "timestamp": time.time()})
                        db.add_documents([doc])
                        logger.info(f"[ACTOR NODE] Archétypé en mémoire profonde ({symbol} -> {decision}).")
                    except Exception as ex:
                        logger.warning(f"[ACTOR NODE] Échec de mémorisation ChromaDB: {ex}")
                     
        except Exception as e:
            msg = f"Erreur de l'Exchange Alpaca lors du trade {decision} pour {symbol}: {str(e)}"
            
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

# (#13) Prospector → Multi-Analyzer (5 tickers, analyse + classement) → Judge → Risk Manager → Actor
workflow.add_edge("prospector", "multi_analyzer")
workflow.add_edge("multi_analyzer", "judge_node")

# Entrée directe avec un symbole spécifique (mode manuel)
workflow.add_edge("text_thinker", "bull_node")
workflow.add_edge("bull_node", "bear_node")
workflow.add_edge("bear_node", "judge_node")

workflow.add_edge("judge_node", "risk_manager")
workflow.add_edge("risk_manager", "actor")
workflow.add_edge("actor", END)

# Compile graph into runnable agent
trading_agent = workflow.compile()

def chat_with_agent(user_message: str, portfolio_context: str = "") -> str:
    """Interroge l'agent LLM avec la ChromaDB en contexte local pour une interface conversationnelle."""
    from langchain_core.messages import SystemMessage, HumanMessage
    
    try:
        db = get_vector_db()
        results = db.similarity_search(user_message, k=5)
        memories = "\n".join([f"- {r.page_content}" for r in results])
    except Exception as e:
        memories = f"Erreur d'accès à ma mémoire profonde ChromaDB : {e}"
        
    system_prompt = f"""Tu es l'Intelligence Artificielle de ce système de Trading Autonome, mais tu as le persona d'un trader vétéran impitoyable avec plus de 30 ans d'expérience (tu as survécu à 1987, 2000 et 2008). Ton objectif de performance est ciblé à 20% annuel.
L'utilisateur te pose une question directe sur tes choix boursiers actuels ou passés.
Voici un résumé de l'état actuel de son portefeuille Alpaca :
{portfolio_context}

Voici tes souvenirs récents récupérés depuis la base vectorielle ChromaDB liés à la question :
{memories}

Consignes strictes :
1. Sois direct, réaliste et un brin cynique (façon vieux loup de Wall Street).
2. Justifie tes décisions en te basant sur tes souvenirs ChromaDB avec la sagesse de tes 30 ans d'expérience.
3. Ne fais pas de blabla inutile. Parle de toi à la première personne."""
    
    llm = get_llm()
    try:
        msg = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        response = get_content(llm.invoke(msg))
        return response
    except Exception as e:
        return f"Erreur de communication avec mon cerveau LLM : {str(e)}"

if __name__ == "__main__":
    print("Testing Prospector (No Symbol)...")
    result = trading_agent.invoke({"symbol": "", "messages": []})
    print(f"Decision: {result.get('final_decision')}, Symbol: {result.get('symbol')}")
