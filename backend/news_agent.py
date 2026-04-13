import os
import re
import math
import time
import asyncio
import logging
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from requests.exceptions import ReadTimeout
from transformers import pipeline, logging as hf_logging
from langchain_community.tools import DuckDuckGoSearchResults

# Silence HF warnings
hf_logging.set_verbosity_error()

load_dotenv()

# --- Structured Logging ---
logger = logging.getLogger("news_agent")
logger.setLevel(logging.INFO)
logging.getLogger("ddgs").setLevel(logging.ERROR)
logging.getLogger("primp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Prevent duplicate handlers if already added by autotrade
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# Global cache for the frontend (replaces the one in agent.py)
LATEST_FINBERT_NEWS = []
MAX_NEWS_CACHE = 20

logger.info("Loading FinBERT model for sentiment analysis (News)...")
try:
    finbert_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, max_length=512)
except Exception as e:
    logger.error(f"Error loading FinBERT: {e}")
    finbert_analyzer = None

class FinancialNewsAgent:
    def __init__(self):
        self.ddg_search = DuckDuckGoSearchResults()

    def get_web_news(self, query: str, config_weight: float = 1.0, max_retries: int = 3) -> List[Dict]:
        """Web search and FinBERT scoring with fault tolerance."""
        news_items = []
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    logger.info(f"DDG: Recherche Web pour '{query}'...")
                
                result = self.ddg_search.invoke(query)
                if result:
                    titles = re.findall(r'title:\s*(.*?),\s*link:', result)
                    for title in titles[:10]:
                        if not title.strip(): continue
                        label, score = self._eval_sentiment(title)
                        
                        final_impact = score * config_weight
                        
                        news_items.append({
                            "source": "Web",
                            "text": title,
                            "label": label,
                            "score": float(score),
                            "weighted_score": float(final_impact)
                        })
                    return news_items # Si ok, on quitte la boucle
            except Exception as e:
                logger.warning(f"DDG: Attempt {attempt+1}/{max_retries} failed for '{query}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff (1s, 2s)
                    
        return news_items

    def _eval_sentiment(self, text: str) -> Tuple[str, float]:
        if not text or finbert_analyzer is None: return "neutral", 0.0
        try:
            res = finbert_analyzer(str(text))[0]
            return res['label'].lower(), res['score']
        except Exception as e:
            return "neutral", 0.0

    async def analyze_market_sentiment_combined(self, symbol: str) -> str:
        """Unified method. Fetches Web News, formats data, caches it, and returns a summary for the agent."""
        
        # We wrap in thread to play nicely with asyncio if needed by the Agent graph
        web_news = await asyncio.to_thread(self.get_web_news, f"{symbol} stock market news", config_weight=1.0)
        
        if not web_news:
            return "No relevant news found."
            
        # Sorted by impact (weighted_score)
        web_news.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        
        summary = f"--- REAL-TIME MARKET INTELLIGENCE (Web) for {symbol} ---\n"
        
        for item in web_news:
            src = item['source']
            lbl = item['label'].upper()
            w_score = item['weighted_score']
            
            summary += f"• [{src} | {lbl}] Title: \"{item['text']}\" (Impact: {w_score:.2f})\n"
                
            # Add to global frontend cache
            news_obj = {
                "title": item['text'][:120] + "..." if len(item['text']) > 120 else item['text'],
                "label": item['label'],
                "score": item['score'],  # pure confidence score for ui
                "source": src,
                "timestamp": time.time()
            }
            LATEST_FINBERT_NEWS.insert(0, news_obj)
            
        # Ensure max length
        while len(LATEST_FINBERT_NEWS) > MAX_NEWS_CACHE:
            LATEST_FINBERT_NEWS.pop()
            
        return summary

# Instances
news_pipeline_agent = FinancialNewsAgent()
