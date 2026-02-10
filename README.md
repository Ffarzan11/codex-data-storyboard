# CSV Storyboard Studio

CSV Storyboard Studio is a local web app that helps users upload CSV files, clean and structure the data, run exploratory analysis, visualize key patterns, and generate a plain-English business story.

## What It Does
- Uploads a user CSV and validates it
- Cleans data (duplicates, missing values, type normalization)
- Removes low-value/noisy fields
- Detects key relationships across fields
- Builds interactive charts (distribution, relationships, trend)
- Produces a plain-English narrative using Gemini (with safe fallback)
- Allows download of the cleaned CSV

## Tech Stack
- Frontend: React + Vite + Framer Motion + Recharts
- Backend: FastAPI + Pandas + NumPy
- Narrative layer: Gemini API (optional, via `.env`)

## Project Structure
- `backend/main.py` API, analysis logic, narrative generation
- `backend/requirements.txt` Python dependencies
- `backend/.env.example` environment template
- `frontend/src/App.jsx` main UI
- `frontend/src/styles.css` styling and animations

## Local Setup

### 1) Backend
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create env file:

Set your key in `backend/.env`:

Start backend:
```bash
uvicorn main:app --reload --port 8000
```

### 2) Frontend
```bash
npm install
npm run dev
```

## Upload Limit
- Max upload size is **5 MB**.
- Larger files return HTTP `413` with a clear error message.

## Current UX Behavior
- Visual charts appear first for quick pattern scanning
- Story section explains fields and findings in plain English
- UI includes lightweight animations while keeping interaction simple


