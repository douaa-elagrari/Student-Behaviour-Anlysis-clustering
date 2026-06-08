# 🎓 Student Cluster Intelligence

Predict a student's behavioural cluster using a **Contrastive Encoder → K-Means** pipeline,
with **Random Forest surrogate + SHAP** explainability — served through a FastAPI/Uvicorn backend
and a polished single-page frontend.

---

## Project Structure

```
student_cluster_app/
├── backend/
│   ├── main.py              ← FastAPI app (prediction + SHAP)
│   ├── requirements.txt
│   └── artifacts/           ← model files (pre-loaded)
│       ├── scaler.pkl
│       ├── kmeans_contrastive.pkl
│       ├── encoder.keras
│       ├── cluster_descriptions.json
│       ├── cluster_profiles.json
│       ├── feature_names.json
│       └── summary.json
└── frontend/
    └── index.html           ← self-contained UI (open in browser)
```

---

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Python 3.9–3.11 recommended** (TensorFlow/Keras compatibility).

### 2. Start the backend

```bash
uvicorn main:app --reload --port 8000
```

You should see:
```
✅ All artifacts loaded.
✅ Surrogate RF accuracy: 0.9xx
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 3. Open the frontend

Just open `frontend/index.html` in your browser — no build step needed.

> The frontend connects to `http://localhost:8000` by default.

---

## How It Works

| Step | Component | What it does |
|------|-----------|--------------|
| 1 | `StandardScaler` | Normalises the 17 input features (z-score) |
| 2 | Contrastive Encoder (Keras 256→128→64) | Maps scaled features into a dense embedding |
| 3 | K-Means (6 clusters) | Assigns cluster from embedding distances |
| 4 | RF Surrogate + SHAP | Trains a Random Forest on original features to replicate cluster labels, then applies TreeExplainer |

### Clusters

| # | Name | Risk |
|---|------|------|
| 0 | High-Achievement but Strained | High |
| 1 | Balanced & Healthy | Low |
| 2 | Disciplined but Struggling | Medium |
| 3 | Leisure-Oriented | Low |
| 4 | Burnout Risk | **Critical** |
| 5 | Productive but Inconsistent | Medium |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Full prediction + SHAP values |
| `GET` | `/health` | Server health check |
| `GET` | `/features` | Feature names & cluster metadata |

### Example `/predict` payload

```json
{
  "age": 20,
  "study_hours": 12,
  "self_study_hours": 7,
  "online_classes_hours": 2.5,
  "social_media_hours": 1,
  "sleep_hours": 4,
  "screen_time_hours": 12,
  "exercise_minutes": 3,
  "caffeine_intake": 158,
  "mental_health_score": 2,
  "focus_index": 4,
  "burnout_level": 9,
  "productivity_score": 4,
  "exam_score": 16,
  "exam_study_hours": 15,
  "stress_level": 9,
  "intervention": 0
}
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Cannot reach backend` | Make sure uvicorn is running: `uvicorn main:app --reload --port 8000` |
| TensorFlow install fails | Use Python 3.10 and `pip install tensorflow==2.15` |
| CORS error in browser | Backend already allows all origins — check uvicorn is on port 8000 |
| Low surrogate accuracy | Normal for small clusters (e.g., cluster 2 has only 2 members); SHAP still meaningful |
