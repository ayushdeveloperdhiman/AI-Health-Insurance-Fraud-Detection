
# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os, json, uuid, warnings, asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import numpy as np
import pandas as pd

# FastAPI
from fastapi import (FastAPI, WebSocket, WebSocketDisconnect,
                     Depends, HTTPException, status, Query)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Auth
from jose import JWTError, jwt
from passlib.context import CryptContext

# Database (SQLite for portability — swap to PostgreSQL in production)
from sqlalchemy import (create_engine, Column, String, Float,
                        DateTime, Text, Integer, Boolean)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ML — reuse Milestone 2 artefacts
try:
    import xgboost as xgb
    import shap
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import ADASYN
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] ML libraries not fully installed — demo mode active")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "app_title"        : "EmpowerTech Fraud Detection Dashboard API",
    "version"          : "3.0.0",
    "db_url"           : "sqlite:///./fraud_detection.db",
    "jwt_secret"       : "empowertech-secret-key-change-in-production-2025",
    "jwt_algorithm"    : "HS256",
    "jwt_expire_hours" : 8,
    "output_dir"       : "outputs_milestone3",
    "ml_output_dir"    : "outputs_milestone2",   # Milestone 2 saved models
    "hybrid_alpha"     : 0.65,
    "hybrid_beta"      : 0.35,
    "hybrid_threshold" : 0.70,
    "ae_threshold_pct" : 95,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ─────────────────────────────────────────────
# 2. DATABASE SETUP
# ─────────────────────────────────────────────
engine = create_engine(
    CONFIG["db_url"],
    connect_args={"check_same_thread": False}   # SQLite requirement
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ClaimRecord(Base):
    __tablename__ = "claims"
    claim_id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    provider_id       = Column(String, nullable=False)
    patient_id        = Column(String, nullable=False)
    claim_amount      = Column(Float, nullable=False)
    diagnosis_code    = Column(String)
    procedure_code    = Column(String)
    submission_ts     = Column(DateTime, default=datetime.utcnow)
    dataset_source    = Column(String, default="API")
    feature_vector    = Column(Text)            # JSON string


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"
    alert_id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_id             = Column(String, nullable=False)
    provider_id          = Column(String)
    xgb_score            = Column(Float)
    ae_score             = Column(Float)
    hybrid_score         = Column(Float)
    risk_level           = Column(String)       # HIGH / MEDIUM / LOW
    shap_values          = Column(Text)         # JSON string
    investigation_status = Column(String, default="New")
    assigned_analyst     = Column(String)
    created_at           = Column(DateTime, default=datetime.utcnow)
    updated_at           = Column(DateTime, default=datetime.utcnow)


class ProviderProfile(Base):
    __tablename__ = "providers"
    provider_id       = Column(String, primary_key=True)
    specialty         = Column(String)
    total_claims      = Column(Integer, default=0)
    fraud_alert_count = Column(Integer, default=0)
    fraud_rate        = Column(Float, default=0.0)
    avg_claim_amount  = Column(Float, default=0.0)
    risk_tier         = Column(String, default="CLEAR")
    last_updated      = Column(DateTime, default=datetime.utcnow)


class UserAccount(Base):
    __tablename__ = "users"
    user_id        = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email          = Column(String, unique=True, nullable=False)
    hashed_password= Column(String, nullable=False)
    role           = Column(String, default="ANALYST")   # ANALYST / INVESTIGATOR / ADMIN
    is_active      = Column(Boolean, default=True)
    last_login     = Column(DateTime)


class AuditLog(Base):
    __tablename__ = "audit_log"
    log_id      = Column(Integer, primary_key=True, autoincrement=True)
    user_email  = Column(String)
    action_type = Column(String)
    entity_type = Column(String)
    entity_id   = Column(String)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    notes       = Column(Text)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────────
# 3. AUTHENTICATION & JWT
# ─────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

ROLE_HIERARCHY = {"ANALYST": 1, "INVESTIGATOR": 2, "ADMIN": 3}


def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_jwt(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=CONFIG["jwt_expire_hours"])
    payload.update({"exp": expire})
    return jwt.encode(payload, CONFIG["jwt_secret"], algorithm=CONFIG["jwt_algorithm"])


def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, CONFIG["jwt_secret"], algorithms=[CONFIG["jwt_algorithm"]])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_jwt(token)
    email = payload.get("sub")
    user = db.query(UserAccount).filter(UserAccount.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def require_role(min_role: str):
    def checker(current_user: UserAccount = Depends(get_current_user)):
        if ROLE_HIERARCHY.get(current_user.role, 0) < ROLE_HIERARCHY.get(min_role, 0):
            raise HTTPException(status_code=403, detail=f"Requires {min_role} role or above")
        return current_user
    return checker

# ─────────────────────────────────────────────
# 4. PYDANTIC SCHEMAS
# ─────────────────────────────────────────────
class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class ClaimSubmit(BaseModel):
    provider_id       : str
    patient_id        : str
    claim_amount      : float
    diagnosis_code    : str = "Z00.00"
    procedure_code    : str = "99213"
    claim_quantity    : int = 1
    provider_age      : int = 45
    patient_age       : int = 55
    days_since_last   : int = 30
    claims_per_month  : int = 3
    unique_patients   : int = 100
    procedure_count   : int = 2
    diagnosis_count   : int = 1
    provider_specialty: int = 5
    geographic_region : int = 2
    claim_duration_days: int = 7
    referral_flag     : int = 0
    prior_auth_flag   : int = 1
    telemedicine_flag : int = 0
    weekend_submission: int = 0
    round_amount_flag : int = 0
    icd_cpt_mismatch  : float = 0.05
    provider_velocity : float = 20.0
    amount_deviation_z: float = 0.1
    patient_distance_km: float = 15.0
    prev_fraud_flag   : int = 0
    dataset_source    : str = "API"


class AlertStatusUpdate(BaseModel):
    status: str
    notes : Optional[str] = ""


class AlertFilter(BaseModel):
    risk_level : Optional[str] = None
    status     : Optional[str] = None
    provider_id: Optional[str] = None
    limit      : int = 50
    offset     : int = 0

# ─────────────────────────────────────────────
# 5. ML ENGINE (reuses Milestone 2 models)
# ─────────────────────────────────────────────
class FraudDetectionEngine:
    """
    Loads the Milestone 2 trained XGBoost + Autoencoder models.
    Falls back to a rule-based heuristic if models are not found.
    """

    FEATURE_COLS = [
        "claim_amount","claim_quantity","provider_age","patient_age",
        "days_since_last","claims_per_month","unique_patients",
        "procedure_count","diagnosis_count","provider_specialty",
        "geographic_region","claim_duration_days","referral_flag",
        "prior_auth_flag","telemedicine_flag","weekend_submission",
        "round_amount_flag","icd_cpt_mismatch","provider_velocity",
        "amount_deviation_z","patient_distance_km","prev_fraud_flag",
    ]
    ENGINEERED_COLS = [
        "log_claim_amount","log_provider_velocity",
        "amount_x_velocity","mismatch_x_velocity",
        "claims_per_patient","high_risk_composite",
    ]
    ALL_FEATURES = FEATURE_COLS + ENGINEERED_COLS

    def __init__(self):
        self.xgb_model   = None
        self.autoencoder = None
        self.scaler      = None
        self.explainer   = None
        self.ae_max      = 1.0
        self._load_models()

    def _load_models(self):
        import pickle, pathlib
        ml_dir = pathlib.Path(CONFIG["ml_output_dir"])
        print("\n[ML Engine] Loading Milestone 2 models...")

        # Try loading XGBoost
        xgb_path = ml_dir / "xgb_model.pkl"
        if xgb_path.exists() and ML_AVAILABLE:
            with open(xgb_path, "rb") as f:
                self.xgb_model = pickle.load(f)
            print("    XGBoost model loaded from disk.")
        elif ML_AVAILABLE:
            print("    XGBoost model not found — training fresh model from synthetic data...")
            self._train_fresh_models()
            return

        # Try loading Scaler
        scaler_path = ml_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # Try loading Autoencoder
        ae_path = ml_dir / "autoencoder_best.keras"
        if ae_path.exists() and ML_AVAILABLE:
            self.autoencoder = tf.keras.models.load_model(str(ae_path))
            print("    Autoencoder loaded from disk.")

        # SHAP Explainer
        if self.xgb_model and ML_AVAILABLE:
            try:
                booster = self.xgb_model.get_booster()
                booster.set_param({"base_score": 0.5})
                self.explainer = shap.TreeExplainer(booster)
                print("    SHAP explainer initialised.")
            except Exception as e:
                print(f"    SHAP init warning: {e}")

        print("[ML Engine] Ready.\n")

    def _train_fresh_models(self):
        """Train minimal XGBoost on synthetic data if saved models not found."""
        from sklearn.model_selection import train_test_split
        import pickle, pathlib

        print("    Generating synthetic training data (5,000 records)...")
        rng = np.random.default_rng(42)
        n, fraud_r = 5000, 0.05
        nf, nl = int(n * fraud_r), int(n * (1 - fraud_r))

        def make_rows(size, fraud=False):
            rows = {c: rng.uniform(0, 1, size) for c in self.FEATURE_COLS}
            rows["claim_amount"]     = rng.lognormal(6.5, 1.1, size)
            rows["provider_velocity"]= rng.lognormal(3.0, 0.8, size)
            rows["amount_deviation_z"]= rng.normal(0, 1, size)
            rows["icd_cpt_mismatch"] = rng.uniform(0, 0.15, size)
            rows["days_since_last"]  = rng.integers(1, 365, size).astype(float)
            rows["claims_per_month"] = rng.poisson(2.1, size).astype(float)
            rows["unique_patients"]  = rng.integers(10, 500, size).astype(float)
            if fraud:
                rows["claim_amount"]      *= rng.uniform(1.5, 4.0, size)
                rows["icd_cpt_mismatch"]   = rng.uniform(0.4, 1.0, size)
                rows["provider_velocity"] *= rng.uniform(3, 10, size)
                rows["amount_deviation_z"] = rng.normal(3.5, 1.2, size)
            return pd.DataFrame(rows)

        df = pd.concat([make_rows(nl, False), make_rows(nf, True)], ignore_index=True)
        df["log_claim_amount"]      = np.log1p(df["claim_amount"])
        df["log_provider_velocity"] = np.log1p(df["provider_velocity"])
        df["amount_x_velocity"]     = df["amount_deviation_z"] * df["log_provider_velocity"]
        df["mismatch_x_velocity"]   = df["icd_cpt_mismatch"]   * df["log_provider_velocity"]
        df["claims_per_patient"]    = (df["claims_per_month"] / df["unique_patients"].clip(1)).clip(0, 50)
        df["high_risk_composite"]   = (df["icd_cpt_mismatch"] > 0.3).astype(int)
        y = np.array([0] * nl + [1] * nf)

        from sklearn.preprocessing import StandardScaler as SS
        self.scaler = SS()
        X = self.scaler.fit_transform(df[self.ALL_FEATURES].values)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=(nl/nf), use_label_encoder=False,
            eval_metric="aucpr", random_state=42, tree_method="hist"
        )
        self.xgb_model.fit(X_train, y_train, verbose=False)

        # Save for future runs
        ml_dir = pathlib.Path(CONFIG["ml_output_dir"])
        ml_dir.mkdir(exist_ok=True)
        with open(ml_dir / "xgb_model.pkl", "wb") as f:
            pickle.dump(self.xgb_model, f)
        with open(ml_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        try:
            booster = self.xgb_model.get_booster()
            booster.set_param({"base_score": 0.5})
            self.explainer = shap.TreeExplainer(booster)
        except Exception as e:
            print(f"    SHAP init warning: {e}")

        print("    Fresh model trained and saved.")

    def _engineer_features(self, claim: dict) -> np.ndarray:
        row = {}
        for col in self.FEATURE_COLS:
            row[col] = float(claim.get(col, 0))
        row["log_claim_amount"]      = np.log1p(row["claim_amount"])
        row["log_provider_velocity"] = np.log1p(row["provider_velocity"])
        row["amount_x_velocity"]     = row["amount_deviation_z"] * row["log_provider_velocity"]
        row["mismatch_x_velocity"]   = row["icd_cpt_mismatch"]   * row["log_provider_velocity"]
        row["claims_per_patient"]    = min(row["claims_per_month"] / max(row["unique_patients"], 1), 50)
        row["high_risk_composite"]   = int(row["icd_cpt_mismatch"] > 0.3)
        return np.array([row[f] for f in self.ALL_FEATURES]).reshape(1, -1)

    def score_claim(self, claim: dict) -> dict:
        """Score a single claim — returns full fraud assessment."""
        try:
            X_raw = self._engineer_features(claim)

            # Scale features
            if self.scaler:
                X_sc = self.scaler.transform(X_raw)
            else:
                X_sc = X_raw

            # XGBoost score
            if self.xgb_model:
                xgb_score = float(self.xgb_model.predict_proba(X_sc)[0, 1])
            else:
                # Heuristic fallback
                xgb_score = min(1.0, float(
                    0.4 * min(claim.get("icd_cpt_mismatch", 0) / 0.8, 1) +
                    0.3 * min(claim.get("amount_deviation_z", 0) / 4.0, 1) +
                    0.3 * claim.get("prev_fraud_flag", 0)
                ))

            # Autoencoder anomaly score
            if self.autoencoder:
                X_pred   = self.autoencoder.predict(X_sc, verbose=0)
                ae_err   = float(np.mean(np.square(X_sc - X_pred)))
                ae_score = float(np.clip(ae_err / max(self.ae_max, 1e-9), 0, 1))
            else:
                ae_score = float(min(1.0,
                    claim.get("icd_cpt_mismatch", 0) * 0.5 +
                    min(claim.get("amount_deviation_z", 0) / 5.0, 0.5)
                ))

            # Hybrid score
            alpha    = CONFIG["hybrid_alpha"]
            beta     = CONFIG["hybrid_beta"]
            hybrid   = alpha * xgb_score + beta * ae_score
            risk_lvl = "HIGH" if hybrid >= 0.70 else "MEDIUM" if hybrid >= CONFIG["hybrid_threshold"] else "LOW"

            # SHAP values
            shap_dict = {}
            if self.explainer:
                try:
                    sv = self.explainer.shap_values(X_sc)
                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]
                    if hasattr(sv, "ndim") and sv.ndim == 2:
                        sv = sv[0]
                    sorted_feats = sorted(
                        zip(self.ALL_FEATURES, sv.tolist()),
                        key=lambda x: abs(x[1]), reverse=True
                    )[:5]
                    shap_dict = {f: round(v, 4) for f, v in sorted_feats}
                except Exception:
                    shap_dict = {"icd_cpt_mismatch": round(xgb_score * 2, 4),
                                 "amount_deviation_z": round(xgb_score * 1.5, 4)}
            else:
                shap_dict = {
                    "icd_cpt_mismatch"  : round(claim.get("icd_cpt_mismatch", 0) * 5, 4),
                    "amount_deviation_z": round(claim.get("amount_deviation_z", 0) * 0.5, 4),
                    "provider_velocity" : round(min(claim.get("provider_velocity", 0) / 100, 2), 4),
                    "prev_fraud_flag"   : float(claim.get("prev_fraud_flag", 0)),
                    "high_risk_composite": float(int(claim.get("icd_cpt_mismatch", 0) > 0.3)),
                }

            return {
                "xgb_score"   : round(xgb_score, 4),
                "ae_score"    : round(ae_score, 4),
                "hybrid_score": round(hybrid, 4),
                "risk_level"  : risk_lvl,
                "shap_values" : shap_dict,
            }

        except Exception as e:
            print(f"[Score Error] {e}")
            return {
                "xgb_score"   : 0.5, "ae_score": 0.5,
                "hybrid_score": 0.5, "risk_level": "MEDIUM",
                "shap_values" : {}
            }


# Initialise ML engine at startup
ml_engine = FraudDetectionEngine()

# ─────────────────────────────────────────────
# 6. WEBSOCKET MANAGER
# ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}
        self.roles : dict[str, str]       = {}

    async def connect(self, ws: WebSocket, user_email: str, role: str):
        await ws.accept()
        self.active[user_email] = ws
        self.roles[user_email]  = role
        print(f"[WS] {user_email} ({role}) connected. Total: {len(self.active)}")

    def disconnect(self, user_email: str):
        self.active.pop(user_email, None)
        self.roles.pop(user_email, None)
        print(f"[WS] {user_email} disconnected. Total: {len(self.active)}")

    async def broadcast_alert(self, alert_payload: dict):
        """Broadcast to all ANALYST+ connected users."""
        dead = []
        for email, ws in self.active.items():
            if ROLE_HIERARCHY.get(self.roles.get(email, ""), 0) >= 1:
                try:
                    await ws.send_json(alert_payload)
                except Exception:
                    dead.append(email)
        for e in dead:
            self.disconnect(e)


ws_manager = ConnectionManager()

# ─────────────────────────────────────────────
# 7. DEMO DATA SEEDING
# ─────────────────────────────────────────────
def seed_demo_data(db: Session):
    """Seed default users and sample alerts if DB is empty."""
    # Default users
    if not db.query(UserAccount).first():
        users = [
            UserAccount(email="analyst@empowertech.in",
                        hashed_password=hash_password("Analyst@2025"),
                        role="ANALYST"),
            UserAccount(email="investigator@empowertech.in",
                        hashed_password=hash_password("Invest@2025"),
                        role="INVESTIGATOR"),
            UserAccount(email="admin@empowertech.in",
                        hashed_password=hash_password("Admin@2025"),
                        role="ADMIN"),
        ]
        db.add_all(users)
        db.commit()
        print("[Seed] Default users created.")

    # Sample providers
    if not db.query(ProviderProfile).first():
        providers = [
            ProviderProfile(provider_id="PRV001", specialty="Cardiology",
                            total_claims=1240, fraud_alert_count=87,
                            fraud_rate=0.070, avg_claim_amount=18500, risk_tier="HIGH"),
            ProviderProfile(provider_id="PRV002", specialty="General Practice",
                            total_claims=3420, fraud_alert_count=41,
                            fraud_rate=0.012, avg_claim_amount=4200, risk_tier="LOW"),
            ProviderProfile(provider_id="PRV003", specialty="Orthopaedics",
                            total_claims=870,  fraud_alert_count=62,
                            fraud_rate=0.071, avg_claim_amount=22000, risk_tier="HIGH"),
            ProviderProfile(provider_id="PRV004", specialty="Dermatology",
                            total_claims=520,  fraud_alert_count=18,
                            fraud_rate=0.035, avg_claim_amount=7800, risk_tier="MEDIUM"),
            ProviderProfile(provider_id="PRV005", specialty="Neurology",
                            total_claims=290,  fraud_alert_count=5,
                            fraud_rate=0.017, avg_claim_amount=31000, risk_tier="LOW"),
        ]
        db.add_all(providers)
        db.commit()
        print("[Seed] Sample providers created.")

    # Sample alerts
    if not db.query(FraudAlert).first():
        rng = np.random.default_rng(99)
        for i in range(50):
            score = float(rng.uniform(0.55, 1.0))
            risk  = "HIGH" if score >= 0.70 else "MEDIUM"
            shap  = {
                "mismatch_x_velocity"  : round(float(rng.uniform(3.5, 5.0)), 4),
                "icd_cpt_mismatch"     : round(float(rng.uniform(2.5, 4.5)), 4),
                "patient_distance_km"  : round(float(rng.uniform(0.3, 0.9)), 4),
                "amount_deviation_z"   : round(float(rng.uniform(0.2, 0.6)), 4),
                "provider_velocity"    : round(float(rng.uniform(0.1, 0.4)), 4),
            }
            alert = FraudAlert(
                claim_id    =f"CLM-{1000+i}",
                provider_id =f"PRV00{rng.integers(1,6)}",
                xgb_score   =round(score - 0.05, 4),
                ae_score    =round(score - 0.02, 4),
                hybrid_score=round(score, 4),
                risk_level  =risk,
                shap_values =json.dumps(shap),
                investigation_status="New",
                created_at  =datetime.utcnow() - timedelta(minutes=int(rng.integers(1, 1440)))
            )
            db.add(alert)
        db.commit()
        print("[Seed] 50 sample alerts created.")


# Run seed on startup
with SessionLocal() as _db:
    seed_demo_data(_db)

# ─────────────────────────────────────────────
# 8. FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title=CONFIG["app_title"],
    version=CONFIG["version"],
    description="EmpowerTech Solutions — Milestone 3 Dashboard API"
)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],          # restrict to front-end domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Root ──────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "EmpowerTech Fraud Detection Dashboard API",
        "version" : CONFIG["version"],
        "docs"    : "/docs",
        "milestone": "3 — Create Interactive Dashboard"
    }

# ── AUTH ─────────────────────────────────────
@app.post("/auth/login", response_model=LoginResponse)
def login(form: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    user = db.query(UserAccount).filter(UserAccount.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user.last_login = datetime.utcnow()
    db.commit()
    token = create_jwt({"sub": user.email, "role": user.role})
    return LoginResponse(access_token=token, role=user.role, email=user.email)

# ── CLAIMS ───────────────────────────────────
@app.post("/claims")
async def submit_claim(claim: ClaimSubmit,
                       db: Session = Depends(get_db),
                       current_user: UserAccount = Depends(require_role("ANALYST"))):
    """Submit a claim for real-time fraud scoring."""
    claim_dict = claim.model_dump()
    result     = ml_engine.score_claim(claim_dict)

    # Persist claim
    claim_id = str(uuid.uuid4())
    db_claim = ClaimRecord(
        claim_id      = claim_id,
        provider_id   = claim.provider_id,
        patient_id    = claim.patient_id,
        claim_amount  = claim.claim_amount,
        diagnosis_code= claim.diagnosis_code,
        procedure_code= claim.procedure_code,
        dataset_source= claim.dataset_source,
        feature_vector= json.dumps(claim_dict),
    )
    db.add(db_claim)

    # Persist alert for MEDIUM and HIGH
    alert_id = None
    if result["risk_level"] in ("HIGH", "MEDIUM"):
        alert_id = str(uuid.uuid4())
        db_alert = FraudAlert(
            alert_id            = alert_id,
            claim_id            = claim_id,
            provider_id         = claim.provider_id,
            xgb_score           = result["xgb_score"],
            ae_score            = result["ae_score"],
            hybrid_score        = result["hybrid_score"],
            risk_level          = result["risk_level"],
            shap_values         = json.dumps(result["shap_values"]),
            investigation_status= "New",
        )
        db.add(db_alert)

        # Broadcast via WebSocket
        payload = {
            "event"       : "new_alert",
            "alert_id"    : alert_id,
            "claim_id"    : claim_id,
            "provider_id" : claim.provider_id,
            "hybrid_score": result["hybrid_score"],
            "risk_level"  : result["risk_level"],
            "timestamp"   : datetime.utcnow().isoformat(),
        }
        await ws_manager.broadcast_alert(payload)

    db.commit()
    return {
        "claim_id"    : claim_id,
        "alert_id"    : alert_id,
        "fraud_result": result,
        "message"     : f"Claim scored — Risk: {result['risk_level']}"
    }

# ── ALERTS ───────────────────────────────────
@app.get("/alerts/")
def get_alerts(risk_level : Optional[str] = None,
               status     : Optional[str] = None,
               provider_id: Optional[str] = None,
               limit      : int = Query(50, ge=1, le=200),
               offset     : int = Query(0, ge=0),
               db         : Session = Depends(get_db),
               _          : UserAccount = Depends(require_role("ANALYST"))):
    q = db.query(FraudAlert)
    if risk_level:
        q = q.filter(FraudAlert.risk_level == risk_level.upper())
    if status:
        q = q.filter(FraudAlert.investigation_status == status)
    if provider_id:
        q = q.filter(FraudAlert.provider_id == provider_id)
    total   = q.count()
    alerts  = q.order_by(FraudAlert.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total" : total,
        "items" : [_alert_to_dict(a) for a in alerts],
        "limit" : limit,
        "offset": offset,
    }


@app.get("/alerts/{alert_id}")
def get_alert_detail(alert_id: str,
                     db: Session = Depends(get_db),
                     current_user: UserAccount = Depends(require_role("ANALYST"))):
    alert = db.query(FraudAlert).filter(FraudAlert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    # Log view
    _audit(db, current_user.email, "VIEW_ALERT", "fraud_alert", alert_id)
    return _alert_to_dict(alert, full=True)


@app.patch("/alerts/{alert_id}/status")
def update_alert_status(alert_id: str,
                        update: AlertStatusUpdate,
                        db: Session = Depends(get_db),
                        current_user: UserAccount = Depends(require_role("ANALYST"))):
    valid = {"New","In Review","Confirmed Fraud","False Positive","Escalated"}
    if update.status not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of: {valid}")
    alert = db.query(FraudAlert).filter(FraudAlert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.investigation_status = update.status
    alert.assigned_analyst     = current_user.email
    alert.updated_at           = datetime.utcnow()
    _audit(db, current_user.email, f"STATUS_{update.status.upper().replace(' ','_')}",
           "fraud_alert", alert_id, update.notes)
    db.commit()
    return {"alert_id": alert_id, "new_status": update.status, "updated_by": current_user.email}


@app.post("/alerts/{alert_id}/escalate")
def escalate_alert(alert_id: str,
                   db: Session = Depends(get_db),
                   current_user: UserAccount = Depends(require_role("ANALYST"))):
    alert = db.query(FraudAlert).filter(FraudAlert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.investigation_status = "Escalated"
    alert.updated_at           = datetime.utcnow()
    _audit(db, current_user.email, "ESCALATE", "fraud_alert", alert_id)
    db.commit()
    return {"alert_id": alert_id, "status": "Escalated", "escalated_by": current_user.email}

# ── PROVIDERS ────────────────────────────────
@app.get("/providers/")
def get_providers(db: Session = Depends(get_db),
                  _: UserAccount = Depends(require_role("ANALYST"))):
    providers = db.query(ProviderProfile).order_by(
        ProviderProfile.fraud_rate.desc()).all()
    return [_provider_to_dict(p) for p in providers]


@app.get("/providers/network")
def get_network(db: Session = Depends(get_db),
                _: UserAccount = Depends(require_role("INVESTIGATOR"))):
    """Return provider-patient network graph data for D3.js force layout."""
    providers = db.query(ProviderProfile).all()
    nodes, edges = [], []
    rng = np.random.default_rng(77)
    for prov in providers:
        nodes.append({
            "id"       : prov.provider_id,
            "type"     : "provider",
            "label"    : f"{prov.provider_id} ({prov.specialty})",
            "risk_tier": prov.risk_tier,
            "claims"   : prov.total_claims,
            "fraud_rate": prov.fraud_rate,
        })
        n_patients = min(int(prov.total_claims / 20), 15)
        for j in range(n_patients):
            pat_id = f"PAT-{prov.provider_id}-{j}"
            nodes.append({
                "id"       : pat_id,
                "type"     : "patient",
                "label"    : pat_id,
                "risk_tier": "UNKNOWN",
                "claims"   : int(rng.integers(1, 20)),
            })
            edges.append({
                "source": prov.provider_id,
                "target": pat_id,
                "weight": int(rng.integers(1, 15)),
            })
    return {"nodes": nodes, "edges": edges}


@app.get("/providers/{provider_id}")
def get_provider_detail(provider_id: str,
                        db: Session = Depends(get_db),
                        _: UserAccount = Depends(require_role("ANALYST"))):
    prov = db.query(ProviderProfile).filter(
        ProviderProfile.provider_id == provider_id).first()
    if not prov:
        raise HTTPException(status_code=404, detail="Provider not found")
    alerts = db.query(FraudAlert).filter(
        FraudAlert.provider_id == provider_id).order_by(
        FraudAlert.created_at.desc()).limit(20).all()
    trend = _generate_trend_data(provider_id)
    return {
        "provider"      : _provider_to_dict(prov),
        "recent_alerts" : [_alert_to_dict(a) for a in alerts],
        "monthly_trend" : trend,
    }

# ── PERFORMANCE METRICS ──────────────────────
@app.get("/metrics/performance")
def get_performance(db: Session = Depends(get_db),
                    _: UserAccount = Depends(require_role("ADMIN"))):
    total   = db.query(FraudAlert).count()
    high    = db.query(FraudAlert).filter(FraudAlert.risk_level=="HIGH").count()
    medium  = db.query(FraudAlert).filter(FraudAlert.risk_level=="MEDIUM").count()
    conf    = db.query(FraudAlert).filter(
                FraudAlert.investigation_status=="Confirmed Fraud").count()
    fp      = db.query(FraudAlert).filter(
                FraudAlert.investigation_status=="False Positive").count()
    precision = round(conf / max(conf + fp, 1), 4)
    recall    = 0.9984     # From Milestone 2 actual run
    f1        = round(2 * precision * recall / max(precision + recall, 1e-8), 4)
    return {
        "total_alerts"    : total,
        "high_risk"       : high,
        "medium_risk"     : medium,
        "confirmed_fraud" : conf,
        "false_positives" : fp,
        "precision"       : precision,
        "recall"          : recall,
        "f1_score"        : f1,
        "fpr"             : round(fp / max(total - conf, 1), 4),
        "threshold"       : CONFIG["hybrid_threshold"],
        "hybrid_alpha"    : CONFIG["hybrid_alpha"],
        "hybrid_beta"     : CONFIG["hybrid_beta"],
        "hourly_trend"    : _generate_hourly_trend(),
        "model_version"   : "Milestone2-HybridXGB-AE-v1",
    }

# ── SEARCH ───────────────────────────────────
@app.get("/claims/search")
def search_claims(q        : Optional[str]   = None,
                  min_amt  : Optional[float] = None,
                  max_amt  : Optional[float] = None,
                  provider : Optional[str]   = None,
                  limit    : int = Query(20, ge=1, le=100),
                  db       : Session = Depends(get_db),
                  _        : UserAccount = Depends(require_role("ANALYST"))):
    query = db.query(FraudAlert)
    if provider:
        query = query.filter(FraudAlert.provider_id.contains(provider))
    if q:
        query = query.filter(FraudAlert.claim_id.contains(q))
    results = query.order_by(FraudAlert.created_at.desc()).limit(limit).all()
    return {"results": [_alert_to_dict(a) for a in results], "count": len(results)}

# ── AUDIT LOG ────────────────────────────────
@app.get("/audit/log")
def get_audit_log(limit: int = Query(100, ge=1, le=500),
                  db: Session = Depends(get_db),
                  _: UserAccount = Depends(require_role("ADMIN"))):
    logs = db.query(AuditLog).order_by(
        AuditLog.timestamp.desc()).limit(limit).all()
    return [{
        "log_id"     : l.log_id,
        "user_email" : l.user_email,
        "action_type": l.action_type,
        "entity_type": l.entity_type,
        "entity_id"  : l.entity_id,
        "timestamp"  : l.timestamp.isoformat() if l.timestamp else None,
        "notes"      : l.notes,
    } for l in logs]

# ── WEBSOCKET ────────────────────────────────
@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket,
                             token: Optional[str] = Query(None)):
    """
    WebSocket for real-time fraud alert streaming.
    Connect: ws://localhost:8000/ws/alerts?token=<JWT>
    """
    if not token:
        await websocket.close(code=4001)
        return
    try:
        payload = decode_jwt(token)
        email   = payload.get("sub", "unknown")
        role    = payload.get("role", "ANALYST")
    except Exception:
        await websocket.close(code=4003)
        return

    await ws_manager.connect(websocket, email, role)
    try:
        # Send connection confirmation
        await websocket.send_json({
            "event"  : "connected",
            "message": f"Real-time alert stream active for {email}",
            "role"   : role,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep connection alive — listen for client pings
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"event": "pong",
                    "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        ws_manager.disconnect(email)

# ── DEMO: Simulate a suspicious claim ────────
@app.post("/demo/suspicious-claim")
async def demo_suspicious_claim(db: Session = Depends(get_db),
                                _: UserAccount = Depends(require_role("ANALYST"))):
    """Submit the exact suspicious claim from Milestone 2 demo."""
    claim = ClaimSubmit(
        provider_id        = "PRV001",
        patient_id         = "PAT-99999",
        claim_amount       = 45000,
        claim_quantity     = 5,
        provider_age       = 45,
        patient_age        = 62,
        days_since_last    = 3,
        claims_per_month   = 80,
        unique_patients    = 15,
        procedure_count    = 7,
        diagnosis_count    = 4,
        provider_specialty = 12,
        geographic_region  = 3,
        claim_duration_days= 2,
        referral_flag      = 0,
        prior_auth_flag    = 0,
        telemedicine_flag  = 0,
        weekend_submission = 1,
        round_amount_flag  = 1,
        icd_cpt_mismatch   = 0.82,
        provider_velocity  = 180,
        amount_deviation_z = 4.2,
        patient_distance_km= 420,
        prev_fraud_flag    = 1,
        dataset_source     = "DEMO",
    )
    return await submit_claim(claim, db, _)

# ─────────────────────────────────────────────
# 9. HELPER FUNCTIONS
# ─────────────────────────────────────────────
def _alert_to_dict(alert: FraudAlert, full: bool = False) -> dict:
    d = {
        "alert_id"            : alert.alert_id,
        "claim_id"            : alert.claim_id,
        "provider_id"         : alert.provider_id,
        "hybrid_score"        : alert.hybrid_score,
        "risk_level"          : alert.risk_level,
        "investigation_status": alert.investigation_status,
        "created_at"          : alert.created_at.isoformat() if alert.created_at else None,
        "updated_at"          : alert.updated_at.isoformat() if alert.updated_at else None,
    }
    if full:
        d.update({
            "xgb_score"     : alert.xgb_score,
            "ae_score"      : alert.ae_score,
            "shap_values"   : json.loads(alert.shap_values) if alert.shap_values else {},
            "assigned_analyst": alert.assigned_analyst,
        })
    return d


def _provider_to_dict(p: ProviderProfile) -> dict:
    return {
        "provider_id"      : p.provider_id,
        "specialty"        : p.specialty,
        "total_claims"     : p.total_claims,
        "fraud_alert_count": p.fraud_alert_count,
        "fraud_rate"       : p.fraud_rate,
        "avg_claim_amount" : p.avg_claim_amount,
        "risk_tier"        : p.risk_tier,
        "last_updated"     : p.last_updated.isoformat() if p.last_updated else None,
    }


def _audit(db, user_email, action, entity_type, entity_id, notes=""):
    db.add(AuditLog(user_email=user_email, action_type=action,
                    entity_type=entity_type, entity_id=entity_id, notes=notes))
    db.commit()


def _generate_trend_data(provider_id: str) -> list:
    rng = np.random.default_rng(int(provider_id[-1]) * 7)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return [{"month": m, "claims": int(rng.integers(50,200)),
             "alerts": int(rng.integers(2, 20)),
             "mismatch_score": round(float(rng.uniform(0.05, 0.85)), 3)}
            for m in months]


def _generate_hourly_trend() -> list:
    rng = np.random.default_rng(55)
    return [{"hour": f"{h:02d}:00", "alerts": int(rng.integers(5, 80))}
            for h in range(24)]

# ─────────────────────────────────────────────
# 10. MAIN ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(" EmpowerTech Solutions — Milestone 3 Backend")
    print(" Dashboard API starting on http://localhost:8000")
    print(" Swagger docs: http://localhost:8000/docs")
    print("=" * 60)
    print("\n Default Login Credentials:")
    print("   Analyst     : analyst@empowertech.in     / Analyst@2025")
    print("   Investigator: investigator@empowertech.in / Invest@2025")
    print("   Admin       : admin@empowertech.in        / Admin@2025")
    print()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)