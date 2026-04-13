# 🚀 Trading Multimodal : AI Hedge Fund Agent

Une plateforme de trading autonome et intelligente exploitant des agents décisionnels (LangGraph), l'analyse de sentiment (FinBERT) et des stratégies quantitatives avancées.

![Visual Dashboard Placeholder](https://via.placeholder.com/800x400.png?text=Trading+Multimodal+Dashboard)

## 🌟 Caractéristiques Principaless

- **Agents Décisionnels (LangGraph)** : Architecture multi-agents avec un "Prospecteur", un "Pessimiste" (Bear), un "Optimiste" (Bull) et un "Juge" final.
- **Analyse de Sentiment (FinBERT)** : Analyse en temps réel des actualités financières pour évaluer l'impact psychologique sur les actifs.
- **Micro & Macroscopie** : Surveillance des tendances à 6 mois couplée à une analyse technique intra-day (RSI, MACD, ATR).
- **Gestion des Risques Pro** : Utilisation de **Trailing Stops** (Stops suiveurs) dynamiques calculés sur la volatilité (ATR).
- **Surveillance Panic Intraday** : Moniteur de sécurité qui scanne les positions toutes les 15 minutes pour protéger le capital.
- **Backtesting Temporel** : Capacité de simuler les décisions des agents sur des dates passées.
- **Dashboard Moderne** : Interface Next.js pour visualiser les positions, les actualités et les débats internes des agents.

## 🏗️ Architecture

Le système repose sur une boucle autonome asynchrone :
1. **Prospecteur** : Identifie les tickers prometteurs via recherche Web.
2. **Analyste Technique** : Calcule les indicateurs (RSI, ATR, MACD) via `pandas-ta`.
3. **Le Débat** : Confrontation d'arguments Bull vs Bear alimentés par le LLM (Google Gemini).
4. **Le Juge** : Arbitrage final basé sur la vision macro et les souvenirs contextuels (ChromaDB).
5. **Exécution** : Passage d'ordres automatiques via l'API Alpaca Markets.

## 🛠️ Installation Rapide

### Pré-requis
- Python 3.10+
- Node.js 18+
- Clés API : Alpaca Markets (Paper) et Google Gemini.

### Installation Automatisée
Exécutez simplement le script de configuration à la racine :
```bash
chmod +x setup.sh
./setup.sh
```

### Configuration Manuelle
1. **Backend** :
   - `cd backend`
   - `python -m venv venv`
   - `source venv/bin/activate` (ou `venv\Scripts\activate` sur Windows)
   - `pip install -r requirements.txt`
   - Créez votre fichier `.env` à partir du `.env.example`.

2. **Frontend** :
   - `cd frontend`
   - `npm install`
   - `npm run dev`

## 🚀 Utilisation

1. Lancez le Backend :
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
2. Accédez au Dashboard : `http://localhost:3000`
3. Pour tester une stratégie passée :
   ```bash
   cd backend
   python backtester.py --symbol AAPL --date 2024-03-15
   ```

## ⚠️ Avertissement
Ce logiciel est destiné à des fins éducatives et de Paper Trading (Simulation). Ne l'utilisez jamais avec de l'argent réel sans avoir pleinement conscience des risques liés au trading de dérivés et d'actions.

---
*Développé avec ❤️ pour un trading plus intelligent.*
