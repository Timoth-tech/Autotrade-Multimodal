import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from dotenv import load_dotenv

load_dotenv()

# --- Optimisation du Logging ---
logger = logging.getLogger("autotrade")
logging.getLogger("ddgs").setLevel(logging.ERROR)
logging.getLogger("primp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx.http2").setLevel(logging.WARNING)

def get_alpaca_client():
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or api_key == "votre_cle_api_alpaca":
        raise ValueError("Les clés Alpaca ne sont pas configurées dans .env")
    return TradingClient(api_key, secret_key, paper=True)

CYCLE_INTERVAL_MARKET_OPEN = 60  # 1 minutes quand le marché est ouvert

# Stockage en mémoire du dernier résultat pour le dashboard
last_auto_result = {"status": "idle", "cycle": 0}
autopilot_enabled = True  # Panic switch
strategy_mode = "balanced"  # "aggressive", "balanced", "conservative"

async def periodic_trade_loop():
    """Boucle autonome continue. Tourne sans arrêt et trade quand le marché est ouvert."""
    cycle = 0
    global last_auto_result
    
    while True:
        cycle += 1
        try:
            client = get_alpaca_client()
            clock = client.get_clock()
            
            if clock.is_open and autopilot_enabled:
                logger.info(f"\n{'='*60}")
                logger.info(f"[CYCLE #{cycle}] Marché OUVERT. Lancement automatique de l'agent...")
                logger.info(f"{'='*60}")
                from agent import trading_agent, update_strategy_from_performance, update_macro_strategy
                
                # 0. Rafraîchissement Stratégique Macro (Toutes les ~8h si cycle = 60min)
                if cycle % 8 == 1:
                    update_macro_strategy()
                    
                # 1. Rétrospective et mise à jour de la stratégie micro
                update_strategy_from_performance()
                
                # 2. Appel autonome de la prospection
                result = trading_agent.invoke({"symbol": "", "messages": []})
                decision = result.get('final_decision', 'UNKNOWN')
                symbol = result.get('symbol', '?')
                
                last_auto_result = {
                    "status": "completed",
                    "cycle": cycle,
                    "decision": decision,
                    "symbol": symbol,
                    "messages": result.get("messages", [])
                }
                
                logger.info(f"[CYCLE #{cycle}] Terminé → {decision} sur {symbol}")
                logger.info(f"Prochain cycle dans {CYCLE_INTERVAL_MARKET_OPEN // 60} minutes.")
                await asyncio.sleep(CYCLE_INTERVAL_MARKET_OPEN)
            else:
                # (#9) Calcul précis du temps avant ouverture
                next_open = clock.next_open
                if next_open and clock.timestamp:
                    wait_seconds = (next_open - clock.timestamp).total_seconds()
                    wait_seconds = min(max(wait_seconds, 60), 3600)  # Entre 1 min et 1h
                    next_str = next_open.strftime("%H:%M UTC")
                    wait_min = int(wait_seconds / 60)
                    logger.info(f"[CYCLE #{cycle}] Marché FERMÉ. Ouverture à {next_str}. Sommeil intelligent : {wait_min} min.")
                    last_auto_result = {"status": "market_closed", "cycle": cycle, "next_open": next_str, "wait_min": wait_min}
                    await asyncio.sleep(wait_seconds)
                else:
                    logger.info(f"[CYCLE #{cycle}] Marché FERMÉ. Horloge unavailable, retry dans 5 min.")
                    last_auto_result = {"status": "market_closed", "cycle": cycle, "next_open": "?"}
                    await asyncio.sleep(300)
        except Exception as e:
            err_msg = str(e)
            if "NameResolutionError" in err_msg or "Max retries exceeded" in err_msg:
                logger.error(f"[CYCLE #{cycle}] Problème réseau / connexion indisponible. Attente...")
                last_auto_result = {"status": "error", "cycle": cycle, "error": "Réseau indisponible"}
            else:
                logger.error(f"[CYCLE #{cycle}] ERREUR boucle autonome: {err_msg}")
                last_auto_result = {"status": "error", "cycle": cycle, "error": err_msg}
                
            await asyncio.sleep(60)
            
            # Use a safe check for autopliot mode since clock might be unbound if it failed instantly
            if not autopilot_enabled:
                logger.info(f"[CYCLE #{cycle}] PANIC MODE ACTIF. Agent en pause.")
                last_auto_result["status"] = "paused"
                await asyncio.sleep(30)

async def emergency_monitor_loop():
    """Surveille les positions ouvertes toutes les 15 minutes pour réagir aux flash-crashes."""
    while True:
        try:
            if not autopilot_enabled:
                await asyncio.sleep(60)
                continue
                
            client = get_alpaca_client()
            clock = client.get_clock()
            if clock.is_open:
                positions = client.get_all_positions()
                for pos in positions:
                    plpc = float(pos.unrealized_intraday_plpc) if pos.unrealized_intraday_plpc else 0.0
                    
                    # Si chute subite de + de 5% dans la journée, on réveille l'agent
                    if plpc < -0.05:
                        logger.warning(f"[PANIC MONITOR] 🚨 Chute anormale ({plpc*100:.1f}%) détectée sur {pos.symbol} ! "
                                       "Lancement d'une délibération d'urgence du Juge...")
                        
                        from agent import trading_agent
                        def run_emergency_agent():
                            trading_agent.invoke({"symbol": pos.symbol})
                        await asyncio.to_thread(run_emergency_agent)
                        
            await asyncio.sleep(900)  # Check toutes les 15 minutes
        except Exception as e:
            logger.error(f"[PANIC MONITOR] Erreur : {e}")
            await asyncio.sleep(300)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop_task, panic_monitor_task
    # Démarrage des tâches de fond
    main_loop_task = asyncio.create_task(periodic_trade_loop())
    panic_monitor_task = asyncio.create_task(emergency_monitor_loop())
    yield
    # Cleanup
    if main_loop_task:
        main_loop_task.cancel()
    if panic_monitor_task:
        panic_monitor_task.cancel()

app = FastAPI(title="Autotrade Multimodal API", lifespan=lifespan)

# Setup CORS to allow Next.js dashboard to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "online", "message": "Autotrade Multimodal Backend is running"}

@app.get("/api/portfolio")
def get_portfolio():
    """Fetch real-time portfolio data from Alpaca Paper Trading"""
    try:
        client = get_alpaca_client()
        account = client.get_account()
        positions = client.get_all_positions()
        
        # Calculate daily PL percentage roughly, if there is Equity
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        change_pct = ((equity - last_equity) / last_equity * 100) if last_equity > 0 else 0.0

        pos_list = []
        for p in positions:
            pos_list.append({
                "symbol": p.symbol, 
                "qty": p.qty, 
                "current_price": float(p.current_price), 
                "unrealized_pl": float(p.unrealized_pl)
            })

        return {
            "status": "success",
            "equity": equity,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "change_pct": round(change_pct, 2),
            "positions": pos_list
        }
    except Exception as e:
        print(f"Alpaca Error: {str(e)}")
        return {
            "status": "error", 
            "message": str(e),
            "equity": 0.0, 
            "buying_power": 0.0, 
            "change_pct": 0.0,
            "positions": []
        }

@app.get("/api/portfolio/analysis")
def get_portfolio_analysis():
    """Trigger the LLM to analyze the current portfolio."""
    from agent import analyze_portfolio_logic
    try:
        analysis = analyze_portfolio_logic()
        return {"status": "success", "analysis": analysis}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# (#10) Endpoint pour le graphique d'équité
@app.get("/api/portfolio/history")
def get_portfolio_history():
    """Retourne l'historique d'équité sur 1 mois pour le graphique."""
    try:
        client = get_alpaca_client()
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(period="1M", timeframe="1D")
        history = client.get_portfolio_history(req)
        
        if history.equity and history.timestamp:
            from datetime import datetime
            data_points = []
            for i, ts in enumerate(history.timestamp):
                data_points.append({
                    "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                    "equity": round(history.equity[i], 2),
                    "profit_loss": round(history.profit_loss[i], 2) if history.profit_loss and i < len(history.profit_loss) else 0
                })
            return {"status": "success", "history": data_points}
        return {"status": "success", "history": []}
    except Exception as e:
        return {"status": "error", "message": str(e), "history": []}

@app.get("/api/agent/status")
def get_agent_status():
    """Retourne le statut actuel de la boucle autonome."""
    return last_auto_result

# --- PANIC SWITCH ---
@app.post("/api/panic")
def panic_switch():
    """Arrêt d'urgence : coupe toutes les positions et désactive l'autopilot."""
    global autopilot_enabled
    try:
        client = get_alpaca_client()
        client.close_all_positions(cancel_orders=True)
        autopilot_enabled = False
        logger.warning("🚨 PANIC SWITCH ACTIVÉ. Toutes positions liquidées. Autopilot désactivé.")
        return {"status": "success", "message": "Toutes les positions ont été liquidées. Autopilot désactivé.", "autopilot_enabled": False}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/panic/resume")
def resume_autopilot():
    """Réactive l'autopilot après un panic switch."""
    global autopilot_enabled
    autopilot_enabled = True
    logger.info("✅ Autopilot réactivé.")
    return {"status": "success", "autopilot_enabled": True}

# --- STRATEGY MODE ---
@app.get("/api/strategy/mode")
def get_strategy_mode():
    return {"mode": strategy_mode, "autopilot_enabled": autopilot_enabled}

@app.post("/api/strategy/mode")
def set_strategy_mode(mode: str = "balanced"):
    global strategy_mode
    if mode not in ["aggressive", "balanced", "conservative"]:
        return {"status": "error", "message": "Modes valides : aggressive, balanced, conservative"}
    strategy_mode = mode
    logger.info(f"Stratégie changée : {mode}")
    return {"status": "success", "mode": strategy_mode}

# --- CONFIDENCE SCORE & SENTIMENT ---
@app.get("/api/confidence")
def get_confidence_score():
    """Score de confiance global + sentiment marché basé sur les données réelles."""
    try:
        client = get_alpaca_client()
        account = client.get_account()
        positions = client.get_all_positions()
        
        # Calcul du score de confiance
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        daily_change = ((equity - last_equity) / last_equity * 100) if last_equity > 0 else 0
        
        # Score basé sur : P&L journalier, nombre de positions gagnantes, buying power
        winning = sum(1 for p in positions if float(p.unrealized_pl) > 0)
        total = len(positions) if positions else 1
        win_rate = winning / total if total > 0 else 0.5
        
        # Score composite (0-100)
        score = min(100, max(0, int(
            50 +  # base
            daily_change * 5 +  # impact du P&L journalier
            win_rate * 30 +  # impact du taux de gain
            (10 if float(account.buying_power) > equity * 0.3 else -10)  # liquidité
        )))
        
        sentiment = "Bullish" if score >= 60 else "Bearish" if score <= 40 else "Neutral"
        
        # Facteurs de décision
        factors = [
            {"name": "Daily P&L", "weight": round(abs(daily_change * 5) / max(score, 1) * 100, 1), "value": f"{daily_change:+.2f}%", "positive": daily_change >= 0},
            {"name": "Win Rate", "weight": round(win_rate * 30 / max(score, 1) * 100, 1), "value": f"{win_rate*100:.0f}%", "positive": win_rate >= 0.5},
            {"name": "Liquidité", "weight": round(10 / max(score, 1) * 100, 1), "value": f"${float(account.buying_power):,.0f}", "positive": float(account.buying_power) > equity * 0.3},
            {"name": "Positions", "weight": round(10 / max(score, 1) * 100, 1), "value": f"{len(positions)}/25", "positive": len(positions) < 25},
        ]
        
        return {
            "status": "success",
            "score": score,
            "sentiment": sentiment,
            "daily_change": round(daily_change, 2),
            "win_rate": round(win_rate * 100, 1),
            "factors": factors,
            "strategy_mode": strategy_mode,
            "autopilot_enabled": autopilot_enabled
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "score": 50, "sentiment": "Unknown"}

# --- DRAWDOWN ---
@app.get("/api/portfolio/drawdown")
def get_drawdown():
    """Calcule le drawdown max historique vs actuel."""
    try:
        client = get_alpaca_client()
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(period="3M", timeframe="1D")
        history = client.get_portfolio_history(req)
        
        if history.equity:
            from datetime import datetime
            equities = history.equity
            peak = equities[0]
            max_dd = 0
            current_dd = 0
            dd_series = []
            
            for i, eq in enumerate(equities):
                if eq > peak:
                    peak = eq
                dd = ((eq - peak) / peak) * 100 if peak > 0 else 0
                if dd < max_dd:
                    max_dd = dd
                dd_series.append({
                    "date": datetime.fromtimestamp(history.timestamp[i]).strftime("%Y-%m-%d"),
                    "drawdown": round(dd, 2)
                })
                current_dd = dd
            
            return {
                "status": "success",
                "max_drawdown": round(max_dd, 2),
                "current_drawdown": round(current_dd, 2),
                "series": dd_series
            }
        return {"status": "success", "max_drawdown": 0, "current_drawdown": 0, "series": []}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- HEATMAP ---
@app.get("/api/portfolio/heatmap")
def get_portfolio_heatmap():
    """Génère les données pour la heatmap du portefeuille."""
    try:
        client = get_alpaca_client()
        positions = client.get_all_positions()
        
        tiles = []
        for p in positions:
            pl_pct = float(p.unrealized_plpc) * 100 if p.unrealized_plpc else 0
            market_value = float(p.market_value) if p.market_value else 0
            tiles.append({
                "symbol": p.symbol,
                "pl_pct": round(pl_pct, 2),
                "market_value": round(market_value, 2),
                "qty": str(p.qty),
                "current_price": float(p.current_price)
            })
        
        return {"status": "success", "tiles": tiles}
    except Exception as e:
        return {"status": "error", "message": str(e), "tiles": []}

@app.get("/api/agent/trigger")
def trigger_agent(symbol: str = ""):
    """Trigger the LangGraph workflow. Empty symbol triggers Autopilot (Prospector)."""
    # We import inside here to avoid circular imports if any, but since we just run agent, it's fine.
    from agent import trading_agent
    try:
        initial_state = {"symbol": symbol, "messages": []}
        result = trading_agent.invoke(initial_state)
        return {
            "status": "success", 
            "decision": result.get("final_decision", "UNKNOWN"), 
            "symbol": result.get("symbol", symbol),
            "messages": result.get("messages", [])
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/agent/debates")
def api_get_debates(request: Request):
    try:
        from agent import GLOBAL_DEBATES_CACHE
        return JSONResponse(content={"debates": GLOBAL_DEBATES_CACHE[::-1]}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/news/sentiment")
def get_news_sentiment():
    """Fetch real-time FinBERT analyzed news from agent cache."""
    from news_agent import LATEST_FINBERT_NEWS
    return {"status": "success", "news": LATEST_FINBERT_NEWS}

from pydantic import BaseModel
class ChatRequest(BaseModel):
    message: str
    portfolio_context: str = ""

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    from agent import chat_with_agent
    try:
        reply = chat_with_agent(req.message, req.portfolio_context)
        return {"status": "success", "reply": reply}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/activities")
def get_activities():
    """Fetch recent execution logs from Alpaca"""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    try:
        client = get_alpaca_client()
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, nested=False)
        orders = client.get_orders(req)
        
        # Format the top activities for the UI (only filled ones)
        data = []
        for o in orders:
            if o.filled_at:
                data.append({
                    "symbol": o.symbol,
                    "side": getattr(o.side, 'value', str(o.side)),
                    "qty": str(o.filled_qty),
                    "price": str(o.filled_avg_price),
                    "time": str(o.filled_at)
                })
            if len(data) >= 10:
                break
        return {"status": "success", "activities": data}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "activities": []}
