"""
Student Cluster Predictor — FastAPI Backend
Pipeline: StandardScaler (16 feats) → Contrastive Encoder (Keras) → KMeans → RF Surrogate SHAP
Run: uvicorn main:app --reload --port 8000
"""

import os, json, warnings
import numpy as np
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")

app = FastAPI(title="Student Cluster Intelligence", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
class State:
    scaler              = None
    kmeans              = None
    encoder             = None
    all_feature_names: list = []   # all 17 (incl. intervention) — for UI & SHAP
    scaler_feature_names: list = []  # only the N features the scaler knows about
    cluster_desc: dict  = {}
    cluster_profiles: dict = {}
    summary: dict       = {}
    surrogate_rf        = None
    shap_explainer      = None

S = State()

# ── Helpers ───────────────────────────────────────────────────────────────────
def scaler_row(feat_dict: dict) -> np.ndarray:
    """Build a (1, N_scaler) array using only the features the scaler was fitted on."""
    return np.array([[feat_dict.get(f, 0.0) for f in S.scaler_feature_names]], dtype=float)

def all_row(feat_dict: dict) -> np.ndarray:
    """Build a (1, 17) array with all features — used for the surrogate RF + SHAP."""
    return np.array([[feat_dict.get(f, 0.0) for f in S.all_feature_names]], dtype=float)

# ── Startup: load + build surrogate ──────────────────────────────────────────
def load_artifacts():
    S.scaler = joblib.load(f"{ARTIFACTS}/scaler.pkl")
    S.kmeans = joblib.load(f"{ARTIFACTS}/kmeans_contrastive.pkl")

    with open(f"{ARTIFACTS}/feature_names.json")        as f: S.all_feature_names = json.load(f)
    with open(f"{ARTIFACTS}/cluster_descriptions.json") as f: S.cluster_desc      = json.load(f)
    with open(f"{ARTIFACTS}/cluster_profiles.json")     as f: S.cluster_profiles  = json.load(f)
    with open(f"{ARTIFACTS}/summary.json")              as f: S.summary           = json.load(f)

    # ── Resolve which features the scaler actually knows ──────────────────────
    # n_features_in_ is set by sklearn >= 1.0 on every fitted transformer
    n_scaler = S.scaler.n_features_in_
    # The scaler was fitted on the features present in cluster_profiles
    # (intervention was added later and is NOT in the scaler)
    profile_feats = [f for f in S.all_feature_names if f in S.cluster_profiles]
    if len(profile_feats) == n_scaler:
        S.scaler_feature_names = profile_feats
    else:
        # Fallback: take the first n_scaler features from all_feature_names
        S.scaler_feature_names = S.all_feature_names[:n_scaler]

    print(f"   Scaler expects {n_scaler} features: {S.scaler_feature_names}")

    import tensorflow as tf
    S.encoder = tf.keras.models.load_model(
        f"{ARTIFACTS}/encoder.keras",
        compile=False,
    )
    print("✅ All artifacts loaded.")


def build_surrogate():
    """
    Synthesise data around each cluster centroid → push through real pipeline
    to get true cluster labels → train surrogate RF on ALL features → SHAP.
    """
    profiles  = S.cluster_profiles
    all_feats = S.all_feature_names
    sc_feats  = S.scaler_feature_names
    n_cl      = int(S.summary["n_clusters"])

    NOISE = {
        "age": 0.6, "study_hours": 1.2, "self_study_hours": 0.7,
        "online_classes_hours": 0.6, "social_media_hours": 0.8,
        "sleep_hours": 0.5, "screen_time_hours": 0.8,
        "exercise_minutes": 8.0, "caffeine_intake": 15.0,
        "mental_health_score": 0.4, "focus_index": 0.4,
        "burnout_level": 0.4, "productivity_score": 0.4,
        "exam_score": 1.2, "exam_study_hours": 1.0,
        "stress_level": 0.4, "intervention": 0.0,
    }

    np.random.seed(42)
    N = 250

    X_all_feats, X_sc_feats, y_all = [], [], []

    for cid in range(n_cl):
        c = str(cid)

        # Centroid for ALL features
        centroid_all = np.array([
            profiles[f][c] if (f in profiles and c in profiles[f]) else 0.0
            for f in all_feats
        ], dtype=float)

        noise_sigma = np.array([NOISE.get(f, 0.5) for f in all_feats])
        samples_all = centroid_all + np.random.randn(N, len(all_feats)) * noise_sigma
        samples_all = np.clip(samples_all, 0, None)

        # Clip binary intervention
        if "intervention" in all_feats:
            idx = all_feats.index("intervention")
            samples_all[:, idx] = np.clip(np.round(samples_all[:, idx]), 0, 1)

        # Subset for scaler (scaler features only)
        sc_indices  = [all_feats.index(f) for f in sc_feats]
        samples_sc  = samples_all[:, sc_indices]

        X_all_feats.append(samples_all)
        X_sc_feats.append(samples_sc)
        y_all.extend([cid] * N)

    X_syn_all = np.vstack(X_all_feats)   # (N*n_cl, 17)  — for surrogate RF
    X_syn_sc  = np.vstack(X_sc_feats)    # (N*n_cl, 16)  — for scaler → encoder → kmeans

    # Ground-truth labels via real pipeline
    X_scaled  = S.scaler.transform(X_syn_sc)
    X_encoded = S.encoder.predict(X_scaled, verbose=0)
    y_true    = S.kmeans.predict(X_encoded)

    # Surrogate RF trained on ALL features (interpretable SHAP space)
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_syn_all, y_true)
    acc = (rf.predict(X_syn_all) == y_true).mean()
    print(f"✅ Surrogate RF accuracy: {acc:.3f}")

    S.surrogate_rf   = rf
    S.shap_explainer = shap.TreeExplainer(rf)


@app.on_event("startup")
async def startup():
    load_artifacts()
    build_surrogate()


# ── Request schema ────────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    age:                  float = 20.0
    study_hours:          float = 6.0
    self_study_hours:     float = 3.0
    online_classes_hours: float = 2.0
    social_media_hours:   float = 3.0
    sleep_hours:          float = 7.0
    screen_time_hours:    float = 8.0
    exercise_minutes:     float = 30.0
    caffeine_intake:      float = 100.0
    mental_health_score:  float = 6.0
    focus_index:          float = 6.0
    burnout_level:        float = 5.0
    productivity_score:   float = 6.0
    exam_score:           float = 14.0
    exam_study_hours:     float = 7.0
    stress_level:         float = 5.0
    intervention:         float = 0.0


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
async def predict(student: StudentInput):
    try:
        feat_dict = student.model_dump()

        # ── Real pipeline (scaler features only) ──────────────────────────────
        x_sc      = scaler_row(feat_dict)          # (1, 16)
        x_scaled  = S.scaler.transform(x_sc)
        x_encoded = S.encoder.predict(x_scaled, verbose=0)
        cluster   = int(S.kmeans.predict(x_encoded)[0])

        # Distance-based cluster probabilities
        dists  = np.linalg.norm(x_encoded - S.kmeans.cluster_centers_, axis=1)
        neg_d  = -dists
        exp_nd = np.exp(neg_d - neg_d.max())
        probs  = (exp_nd / exp_nd.sum()).tolist()

        # ── SHAP — surrogate RF on all 17 features ────────────────────────────
        x_all     = all_row(feat_dict)             # (1, 17)
        shap_vals = S.shap_explainer.shap_values(x_all)
        sv_cluster = (
            shap_vals[cluster][0].tolist()
            if isinstance(shap_vals, list)
            else shap_vals[0, :, cluster].tolist()
        )
        base_val = (
            float(S.shap_explainer.expected_value[cluster])
            if hasattr(S.shap_explainer.expected_value, "__len__")
            else float(S.shap_explainer.expected_value)
        )

        # Cluster centroid (all features)
        cid      = str(cluster)
        centroid = [
            S.cluster_profiles[f][cid]
            if (f in S.cluster_profiles and cid in S.cluster_profiles[f]) else 0.0
            for f in S.all_feature_names
        ]

        return {
            "cluster_id":        cluster,
            "cluster_info":      S.cluster_desc[cid],
            "confidence":        probs[cluster],
            "all_probabilities": probs,
            "shap_values":       sv_cluster,
            "base_value":        base_val,
            "feature_names":     S.all_feature_names,
            "input_values":      x_all[0].tolist(),
            "cluster_centroid":  centroid,
            "all_cluster_info":  S.cluster_desc,
            "summary":           S.summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "loaded": S.scaler is not None,
            "scaler_features": S.scaler_feature_names}

@app.get("/features")
async def features():
    return {
        "feature_names":       S.all_feature_names,
        "scaler_feature_names": S.scaler_feature_names,
        "cluster_info":        S.cluster_desc,
        "summary":             S.summary,
    }
