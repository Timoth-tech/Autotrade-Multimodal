#!/bin/bash

# Automated installation script for Multimodal Trading

echo "🚀 Starting Multimodal Trading installation..."

# 1. Backend Verification
echo "📦 Configuring Backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created."
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Python dependencies installed."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  .env file created from example. Don't forget to add your API keys!"
fi

cd ..

# 2. Frontend Verification
echo "💻 Configuring Frontend..."
if [ -d "frontend" ]; then
    cd frontend
    npm install
    echo "✅ Node.js dependencies installed."
    cd ..
else
    echo "❌ Frontend directory not found."
fi

echo "===================================================="
echo "🎉 Installation completed successfully!"
echo "===================================================="
echo "To launch the system:"
echo "1. Backend: cd backend && uvicorn main:app --reload"
echo "2. Frontend: cd frontend && npm run dev"
echo "===================================================="
