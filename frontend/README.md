# AutoTrade Multimodal

Ce projet contient l'interface utilisateur (frontend Next.js) et l'agent de trading (backend FastAPI) pour le système de trading autonome.

## 🚀 Comment lancer le programme

Pour que l'application fonctionne correctement, vous devez lancer **à la fois** le backend et le frontend dans deux terminaux séparés.

### 1. Lancer le Backend (Python / FastAPI)

Ouvrez un terminal et placez-vous dans le dossier `backend` :

```bash
cd backend

# Optionnel mais recommandé : Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # macOS / Linux
# sur Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur backend
uvicorn main:app --reload
```
Le backend sera alors actif sur http://localhost:8000.

### 2. Lancer le Frontend (Next.js)

Ouvrez un **nouveau** terminal et placez-vous dans le dossier `frontend` :

```bash
cd frontend

# Installer les dépendances (nécessaire à la première utilisation)
npm install

# Lancer le serveur de développement
npm run dev
```

Une fois lancé, ouvrez votre navigateur et accédez à l'adresse **[http://localhost:3000](http://localhost:3000)** pour voir le tableau de bord.
