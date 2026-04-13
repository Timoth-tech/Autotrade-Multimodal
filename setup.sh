#!/bin/bash

# Script d'installation automatisé pour Trading Multimodal

echo "🚀 Démarrage de l'installation de Trading Multimodal..."

# 1. Vérification du Backend
echo "📦 Configuration du Backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Environnement virtuel créé."
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dépendances Python installées."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  Fichier .env créé à partir de l'exemple. N'oubliez pas d'ajouter vos clés API !"
fi

cd ..

# 2. Vérification du Frontend
echo "💻 Configuration du Frontend..."
if [ -d "frontend" ]; then
    cd frontend
    npm install
    echo "✅ Dépendances Node.js installées."
    cd ..
else
    echo "❌ Dossier frontend introuvable."
fi

echo "===================================================="
echo "🎉 Installation terminée avec succès !"
echo "===================================================="
echo "Pour lancer le système :"
echo "1. Backend : cd backend && uvicorn main:app --reload"
echo "2. Frontend : cd frontend && npm run dev"
echo "===================================================="
