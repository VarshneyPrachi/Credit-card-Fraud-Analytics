# FraudGuard API

## Setup (Windows PowerShell)

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Edit `.env` with your own local DB credentials. Never upload `.env` to GitHub. API docs: `http://127.0.0.1:8000/docs`.

`GET /api/health` confirms the service is running. `GET /api/dashboard/summary` reads counts from the `transactions` table. `GET /api/models/performance` reads `vw_model_performance`.

`POST /api/predict` intentionally returns HTTP 501 until the exact training feature schema, preprocessing pipeline, and saved model artifact paths are verified. The original Kaggle dataset model generally needs `Time`, `Amount`, and `V1`–`V28`; amount/time alone must not be represented as a real prediction.
