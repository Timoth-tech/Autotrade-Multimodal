# AutoTrade Multimodal

This project contains the user interface (Next.js frontend) and the trading agent (FastAPI backend) for the autonomous trading system.

## 🚀 How to launch the program

For the application to work correctly, you must launch **both** the backend and the frontend in two separate terminals.

### 1. Launch the Backend (Python / FastAPI)

Open a terminal and navigate to the `backend` folder:

```bash
cd backend

# Optional but recommended: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS / Linux
# on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the backend server
uvicorn main:app --reload
```
The backend will then be active on http://localhost:8000.

### 2. Launch the Frontend (Next.js)

Open a **new** terminal and navigate to the `frontend` folder:

```bash
cd frontend

# Install dependencies (necessary for the first use)
npm install

# Launch the development server
npm run dev
```

Once launched, open your browser and access **[http://localhost:3000](http://localhost:3000)** to see the dashboard.
