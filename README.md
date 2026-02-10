# CSV Storyboard Studio

A local web app that uploads CSV files, cleans/formats data, runs EDA, generates charts, builds a storyboard, and provides a conclusion.

## Stack
- Frontend: React (Vite), Framer Motion, Recharts
- Backend: FastAPI, Pandas, NumPy

## Run Locally

### 1) Backend (port 8000)
```bash
cd "/Users/farhikhtafarzan/Documents/New project/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2) Frontend (port 5173)
```bash
cd "/Users/farhikhtafarzan/Documents/New project/frontend"
npm install
npm run dev
```

Then open: http://127.0.0.1:5173

## API
- `GET /health`
- `POST /analyze` (multipart form with `file` as `.csv`)

## Notes
- Backend auto-cleans duplicate rows, infers some numeric/date columns, and imputes missing values.
- EDA returns histogram(s), scatter relation, and time trend when date fields are detected.
