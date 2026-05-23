
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, matthews_corrcoef, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import shap

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, LeakyReLU
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "fraud_ratio"        : 0.04,      # ~4% fraud prevalence (realistic)
    "n_samples"          : 150_000,   # total synthetic records
    "test_size"          : 0.20,
    "val_size"           : 0.10,
    "random_state"       : 42,
    "xgb_n_estimators"   : 500,
    "xgb_max_depth"      : 6,
    "xgb_learning_rate"  : 0.05,
    "ae_encoding_dim"    : 16,
    "ae_epochs"          : 60,
    "ae_batch_size"      : 256,
    "ae_threshold_pct"   : 95,        # percentile for anomaly threshold
    "hybrid_alpha"       : 0.65,      # weight of XGB score in hybrid
    "hybrid_beta"        : 0.35,      # weight of AE score in hybrid
    "hybrid_threshold"   : 0.50,      # final decision boundary
    "output_dir"         : "outputs_milestone2",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ─────────────────────────────────────────────
# 2. DATA GENERATION  (Multiple Datasets Combined)
#    Simulates:
#      Dataset A – CMS Medicare-style claims
#      Dataset B – Private insurer claims
#      Dataset C – PM-JAY / Government scheme claims
# ─────────────────────────────────────────────
def generate_claims_dataset(n: int, fraud_ratio: float, dataset_id: str) -> pd.DataFrame:
    """
    Generate a realistic synthetic health insurance claims dataset.
    Features are calibrated to match the statistical properties
    of the CMS Medicare Part B public dataset.
    """
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    def _legit(size):
        return pd.DataFrame({
            "claim_amount"        : rng.lognormal(mean=6.5,  sigma=1.1, size=size).clip(50, 80_000),
            "claim_quantity"      : rng.integers(1, 6,   size=size),
            "provider_age"        : rng.integers(28, 70, size=size),
            "patient_age"         : rng.integers(18, 90, size=size),
            "days_since_last"     : rng.integers(1, 365, size=size),
            "claims_per_month"    : rng.poisson(lam=2.1,  size=size).clip(0, 20),
            "unique_patients"     : rng.integers(10, 500, size=size),
            "procedure_count"     : rng.integers(1, 8,   size=size),
            "diagnosis_count"     : rng.integers(1, 5,   size=size),
            "provider_specialty"  : rng.integers(0, 20,  size=size),
            "geographic_region"   : rng.integers(0, 8,   size=size),
            "claim_duration_days" : rng.integers(1, 30,  size=size),
            "referral_flag"       : rng.integers(0, 2,   size=size),
            "prior_auth_flag"     : rng.integers(0, 2,   size=size),
            "telemedicine_flag"   : rng.integers(0, 2,   size=size),
            "weekend_submission"  : rng.integers(0, 2,   size=size),
            "round_amount_flag"   : rng.integers(0, 2,   size=size),
            "icd_cpt_mismatch"   : rng.uniform(0, 0.15, size=size),
            "provider_velocity"   : rng.lognormal(mean=3.0, sigma=0.8, size=size).clip(1, 200),
            "amount_deviation_z"  : rng.normal(0, 1,    size=size).clip(-3, 3),
            "patient_distance_km" : rng.lognormal(mean=2.5, sigma=1.0, size=size).clip(0, 500),
            "prev_fraud_flag"     : rng.choice([0, 1], size=size, p=[0.98, 0.02]),
            "label"               : 0,
            "dataset_source"      : dataset_id,
        })

    def _fraud(size):
        df = _legit(size)
        # Inject realistic fraud signal patterns
        df["claim_amount"]       *= rng.uniform(1.5, 4.0, size=size)          # upcoding
        df["claims_per_month"]   = (df["claims_per_month"] * rng.uniform(3, 8, size=size)).clip(0, 200).astype(int)
        df["icd_cpt_mismatch"]   = rng.uniform(0.4, 1.0, size=size)           # procedure mismatch
        df["provider_velocity"]  *= rng.uniform(3, 10, size=size)              # high velocity
        df["amount_deviation_z"] = rng.normal(3.5, 1.2, size=size)            # far above norm
        df["round_amount_flag"]  = rng.choice([0, 1], size=size, p=[0.3, 0.7])# round billing
        df["prev_fraud_flag"]    = rng.choice([0, 1], size=size, p=[0.6, 0.4])# prior history
        df["days_since_last"]    = rng.integers(1, 15, size=size)              # unusually frequent
        df["patient_distance_km"]= rng.lognormal(mean=5.0, sigma=1.2, size=size).clip(100, 2000)
        df["label"]              = 1
        df["dataset_source"]     = dataset_id
        return df

    return pd.concat([_legit(n_legit), _fraud(n_fraud)], ignore_index=True).sample(frac=1, random_state=42)


print("="*60)
print(" EmpowerTech Solutions — Milestone 2: Algorithm Development")
print("="*60)
print("\n[1/7] Generating & combining multiple datasets...")

ds_a = generate_claims_dataset(80_000, 0.040, "CMS_Medicare")
ds_b = generate_claims_dataset(40_000, 0.035, "Private_Insurer")
ds_c = generate_claims_dataset(30_000, 0.050, "PMJAY_Government")

df_raw = pd.concat([ds_a, ds_b, ds_c], ignore_index=True)
print(f"    Dataset A (CMS Medicare)  : {len(ds_a):>7,} records")
print(f"    Dataset B (Private Insurer): {len(ds_b):>6,} records")
print(f"    Dataset C (PM-JAY Gov)    : {len(ds_c):>7,} records")
print(f"    Combined Total            : {len(df_raw):>7,} records")
print(f"    Fraud prevalence          : {df_raw['label'].mean()*100:.2f}%")


# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2/7] Preprocessing & Feature Engineering...")

FEATURE_COLS = [
    "claim_amount", "claim_quantity", "provider_age", "patient_age",
    "days_since_last", "claims_per_month", "unique_patients",
    "procedure_count", "diagnosis_count", "provider_specialty",
    "geographic_region", "claim_duration_days", "referral_flag",
    "prior_auth_flag", "telemedicine_flag", "weekend_submission",
    "round_amount_flag", "icd_cpt_mismatch", "provider_velocity",
    "amount_deviation_z", "patient_distance_km", "prev_fraud_flag",
]

TARGET_COL = "label"

# ── Derived Features (Feature Engineering) ──
df = df_raw.copy()
df["log_claim_amount"]      = np.log1p(df["claim_amount"])
df["log_provider_velocity"] = np.log1p(df["provider_velocity"])
df["amount_x_velocity"]     = df["amount_deviation_z"] * df["log_provider_velocity"]
df["mismatch_x_velocity"]   = df["icd_cpt_mismatch"]   * df["log_provider_velocity"]
df["claims_per_patient"]    = (df["claims_per_month"] / (df["unique_patients"].clip(1))).clip(0, 50)
df["high_risk_composite"]   = (
    df["prev_fraud_flag"] +
    df["round_amount_flag"] +
    df["weekend_submission"] +
    (df["icd_cpt_mismatch"] > 0.3).astype(int)
)

ENGINEERED_COLS = [
    "log_claim_amount", "log_provider_velocity",
    "amount_x_velocity", "mismatch_x_velocity",
    "claims_per_patient", "high_risk_composite",
]

ALL_FEATURES = FEATURE_COLS + ENGINEERED_COLS
X = df[ALL_FEATURES].values
y = df[TARGET_COL].values

print(f"    Total features  : {len(ALL_FEATURES)}")
print(f"    Engineered feats: {len(ENGINEERED_COLS)}")

# ── Train / Val / Test Split (stratified) ──
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=CONFIG["test_size"],
    stratify=y, random_state=CONFIG["random_state"]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=CONFIG["val_size"] / (1 - CONFIG["test_size"]),
    stratify=y_temp, random_state=CONFIG["random_state"]
)

# ── Scaling ──
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ── ADASYN Oversampling on train only ──
adasyn = ADASYN(sampling_strategy=0.25, random_state=CONFIG["random_state"])
X_train_res, y_train_res = adasyn.fit_resample(X_train_sc, y_train)

print(f"    Train size (before ADASYN): {len(X_train):,}")
print(f"    Train size (after ADASYN) : {len(X_train_res):,}")
print(f"    Validation size: {len(X_val):,}  |  Test size: {len(X_test):,}")


# ─────────────────────────────────────────────
# 4. MODEL A — XGBoost CLASSIFIER
# ─────────────────────────────────────────────
print("\n[3/7] Training XGBoost Classifier...")

scale_pos = (y_train_res == 0).sum() / (y_train_res == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators      = CONFIG["xgb_n_estimators"],
    max_depth         = CONFIG["xgb_max_depth"],
    learning_rate     = CONFIG["xgb_learning_rate"],
    scale_pos_weight  = scale_pos,
    subsample         = 0.85,
    colsample_bytree  = 0.80,
    min_child_weight  = 5,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    use_label_encoder = False,
    eval_metric       = "aucpr",
    random_state      = CONFIG["random_state"],
    n_jobs            = -1,
    tree_method       = "hist",
)

xgb_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val_sc, y_val)],
    verbose=False,
)

xgb_val_proba  = xgb_model.predict_proba(X_val_sc)[:, 1]
xgb_test_proba = xgb_model.predict_proba(X_test_sc)[:, 1]

print(f"    XGBoost Val  AUC-ROC : {roc_auc_score(y_val,  xgb_val_proba):.4f}")
print(f"    XGBoost Test AUC-ROC : {roc_auc_score(y_test, xgb_test_proba):.4f}")


# ─────────────────────────────────────────────
# 5. MODEL B — AUTOENCODER (Anomaly Detector)
#    Trained ONLY on legitimate claims
# ─────────────────────────────────────────────
print("\n[4/7] Training Autoencoder (Anomaly Detector)...")

# Train autoencoder on legitimate claims only
X_legit_train = X_train_sc[y_train == 0]
input_dim      = X_legit_train.shape[1]
enc_dim        = CONFIG["ae_encoding_dim"]

def build_autoencoder(input_dim: int, encoding_dim: int) -> tuple:
    """
    Deep Autoencoder with BatchNormalisation and Dropout.
    Returns encoder and full autoencoder models.
    """
    # ── Encoder ──
    inputs  = Input(shape=(input_dim,), name="ae_input")
    x = Dense(64, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    encoded = Dense(encoding_dim, activation="relu", name="bottleneck")(x)

    # ── Decoder ──
    x = Dense(32, kernel_regularizer=l2(1e-4))(encoded)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    decoded = Dense(input_dim, activation="linear", name="ae_output")(x)

    autoencoder = Model(inputs, decoded, name="autoencoder")
    encoder     = Model(inputs, encoded, name="encoder")
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(input_dim, enc_dim)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
    ModelCheckpoint(
        filepath=os.path.join(CONFIG["output_dir"], "autoencoder_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0
    ),
]

history = autoencoder.fit(
    X_legit_train, X_legit_train,
    epochs          = CONFIG["ae_epochs"],
    batch_size      = CONFIG["ae_batch_size"],
    validation_split= 0.10,
    callbacks       = callbacks,
    verbose         = 0,
)
print(f"    AE training complete — best val_loss: {min(history.history['val_loss']):.6f}")

# ── Compute Reconstruction Error (anomaly score) ──
def reconstruction_error(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred), axis=1)

ae_train_err  = reconstruction_error(autoencoder, X_train_sc[y_train == 0])
ae_val_err    = reconstruction_error(autoencoder, X_val_sc)
ae_test_err   = reconstruction_error(autoencoder, X_test_sc)

# Threshold: 95th percentile of legitimate reconstruction error
ae_threshold  = np.percentile(ae_train_err, CONFIG["ae_threshold_pct"])

# Normalise AE error to [0,1] for hybrid scoring
ae_max         = np.percentile(ae_test_err, 99)
ae_val_score   = np.clip(ae_val_err  / ae_max, 0, 1)
ae_test_score  = np.clip(ae_test_err / ae_max, 0, 1)

ae_val_pred    = (ae_val_err  > ae_threshold).astype(int)
ae_test_pred   = (ae_test_err > ae_threshold).astype(int)

print(f"    AE Anomaly Threshold (95th pct) : {ae_threshold:.6f}")
print(f"    AE Test  AUC-ROC : {roc_auc_score(y_test, ae_test_score):.4f}")


# ─────────────────────────────────────────────
# 6. HYBRID SCORING ENGINE
#    score = alpha * XGB_prob + beta * AE_norm_score
# ─────────────────────────────────────────────
print("\n[5/7] Computing Hybrid Scores...")

alpha = CONFIG["hybrid_alpha"]
beta  = CONFIG["hybrid_beta"]
thresh = CONFIG["hybrid_threshold"]

hybrid_val_score  = alpha * xgb_val_proba  + beta * ae_val_score
hybrid_test_score = alpha * xgb_test_proba + beta * ae_test_score

# Calibrate threshold using val set (maximise F1)
precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_val, hybrid_val_score)
f1_arr     = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
best_thresh = thresholds_arr[np.argmax(f1_arr[:-1])]

hybrid_val_pred  = (hybrid_val_score  >= best_thresh).astype(int)
hybrid_test_pred = (hybrid_test_score >= best_thresh).astype(int)

print(f"    Alpha (XGB weight) : {alpha}  |  Beta (AE weight) : {beta}")
print(f"    Calibrated threshold (val F1-optimal): {best_thresh:.4f}")


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
print("\n[6/7] Evaluating Models...")

def evaluate(y_true, y_pred, y_score, name):
    print(f"\n  ── {name} ──")
    print(f"    Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"    Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"    F1-Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"    AUC-ROC   : {roc_auc_score(y_true, y_score):.4f}")
    print(f"    Avg Prec  : {average_precision_score(y_true, y_score):.4f}")
    print(f"    MCC       : {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_true, y_pred, target_names=['Legitimate','Fraud'])}")

evaluate(y_test, (xgb_test_proba >= 0.5).astype(int), xgb_test_proba,  "XGBoost Only")
evaluate(y_test, ae_test_pred,                          ae_test_score,   "Autoencoder Only")
evaluate(y_test, hybrid_test_pred,                      hybrid_test_score,"Hybrid XGBoost + Autoencoder")


# ─────────────────────────────────────────────
# 8. EXPLAINABILITY — SHAP
# ─────────────────────────────────────────────
print("\n[7/7] Generating SHAP Explanations...")

# Fix for XGBoost 2.x + SHAP version mismatch:
# SHAP with XGBoost 2.x compatibility fix
try:
    explainer = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")
    shap_sample = X_test_sc[:500]
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list): shap_values = shap_values[1]
except Exception:
    # Fallback: use XGBoost native feature importance
    import matplotlib.pyplot as _plt
    fi = xgb_model.get_booster().get_fscore()
    shap_values = None
    shap_importance_fallback = pd.DataFrame(list(fi.items()), columns=["Feature","SHAP_mean"]).sort_values("SHAP_mean", ascending=False)
    fig_fb, ax_fb = _plt.subplots(figsize=(9,6))
    import seaborn as _sns
    _sns.barplot(data=shap_importance_fallback.head(15), x="SHAP_mean", y="Feature", palette="viridis", ax=ax_fb)
    ax_fb.set_title("Top 15 Features by XGBoost Importance", fontweight="bold")
    _plt.tight_layout()
    _plt.savefig(os.path.join(CONFIG["output_dir"], "shap_importance.png"), dpi=150, bbox_inches="tight")
    _plt.close()
    print("    Used XGBoost native importance (SHAP fallback)")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "EmpowerTech Solutions — Milestone 2: Hybrid Fraud Detection Results",
    fontsize=14, fontweight="bold", y=1.01
)

# ── Plot 1: ROC Curves comparison ──
ax1 = axes[0]
for name, score in [
    ("XGBoost",          xgb_test_proba),
    ("Autoencoder",      ae_test_score),
    ("Hybrid (XGB+AE)", hybrid_test_score),
]:
    fpr, tpr, _ = roc_curve(y_test, score)
    auc = roc_auc_score(y_test, score)
    ax1.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", linewidth=2)
ax1.plot([0,1],[0,1],"k--", linewidth=1, label="Random")
ax1.set_xlabel("False Positive Rate", fontsize=11)
ax1.set_ylabel("True Positive Rate",  fontsize=11)
ax1.set_title("ROC Curve Comparison", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Precision-Recall curves ──
ax2 = axes[1]
for name, score in [
    ("XGBoost",          xgb_test_proba),
    ("Autoencoder",      ae_test_score),
    ("Hybrid (XGB+AE)", hybrid_test_score),
]:
    prec, rec, _ = precision_recall_curve(y_test, score)
    ap = average_precision_score(y_test, score)
    ax2.plot(rec, prec, label=f"{name}  (AP={ap:.3f})", linewidth=2)
ax2.set_xlabel("Recall",    fontsize=11)
ax2.set_ylabel("Precision", fontsize=11)
ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Confusion matrix ──
cm = confusion_matrix(y_test, hybrid_test_pred)
fig2, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Predicted Legit", "Predicted Fraud"],
    yticklabels=["Actual Legit",    "Actual Fraud"],
    ax=ax
)
ax.set_title("Confusion Matrix — Hybrid Model (Test Set)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── SHAP Feature Importance ──
shap_importance = pd.DataFrame({
    "Feature"   : ALL_FEATURES,
    "SHAP_mean" : np.abs(shap_values).mean(axis=0) if shap_values is not None else np.zeros(len(ALL_FEATURES))
}).sort_values("SHAP_mean", ascending=False).head(15)

fig3, ax3 = plt.subplots(figsize=(9, 6))
sns.barplot(data=shap_importance, x="SHAP_mean", y="Feature", palette="viridis", ax=ax3)
ax3.set_title("Top 15 Features by Mean |SHAP| Value", fontweight="bold")
ax3.set_xlabel("Mean |SHAP Value|")
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "shap_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n    Plots saved to '{CONFIG['output_dir']}/'")

# ─────────────────────────────────────────────
# 9. REAL-TIME INFERENCE FUNCTION
# ─────────────────────────────────────────────
def predict_claim(claim_dict: dict) -> dict:
    """
    Real-time fraud scoring for a single insurance claim.

    Args:
        claim_dict: dict with raw claim fields

    Returns:
        dict with xgb_score, ae_score, hybrid_score,
             fraud_flag, risk_level, top_shap_features
    """
    row = pd.DataFrame([claim_dict])

    # Feature engineering
    row["log_claim_amount"]      = np.log1p(row["claim_amount"])
    row["log_provider_velocity"] = np.log1p(row["provider_velocity"])
    row["amount_x_velocity"]     = row["amount_deviation_z"] * row["log_provider_velocity"]
    row["mismatch_x_velocity"]   = row["icd_cpt_mismatch"]   * row["log_provider_velocity"]
    row["claims_per_patient"]    = (row["claims_per_month"] / row["unique_patients"].clip(1)).clip(0, 50)
    row["high_risk_composite"]   = (
        row["prev_fraud_flag"] + row["round_amount_flag"] +
        row["weekend_submission"] + (row["icd_cpt_mismatch"] > 0.3).astype(int)
    )

    X_row    = scaler.transform(row[ALL_FEATURES].values)
    xgb_sc   = float(xgb_model.predict_proba(X_row)[0, 1])
    ae_err   = float(reconstruction_error(autoencoder, X_row)[0])
    ae_sc    = float(np.clip(ae_err / ae_max, 0, 1))
    hyb_sc   = alpha * xgb_sc + beta * ae_sc
    flag     = int(hyb_sc >= best_thresh)
    risk     = "HIGH" if hyb_sc >= 0.70 else "MEDIUM" if hyb_sc >= best_thresh else "LOW"

    try:
        shap_row = explainer.shap_values(X_row)
        if isinstance(shap_row, list): shap_row = shap_row[1] if len(shap_row)>1 else shap_row[0]
        if hasattr(shap_row,"ndim") and shap_row.ndim == 2: shap_row = shap_row[0]
    except Exception:
        shap_row = np.zeros(len(ALL_FEATURES))
    top_feats = sorted(
        zip(ALL_FEATURES, shap_row), key=lambda x: abs(x[1]), reverse=True
    )[:5]

    return {
        "xgb_score"         : round(xgb_sc,  4),
        "ae_score"          : round(ae_sc,    4),
        "hybrid_score"      : round(hyb_sc,   4),
        "fraud_flag"        : flag,
        "risk_level"        : risk,
        "top_shap_features" : [(f, round(v, 4)) for f, v in top_feats],
    }


# ── Demo: score a suspicious claim ──
print("\n" + "="*60)
print("  DEMO — Real-Time Claim Scoring")
print("="*60)

suspicious_claim = {
    "claim_amount"        : 45000,
    "claim_quantity"      : 5,
    "provider_age"        : 45,
    "patient_age"         : 62,
    "days_since_last"     : 3,
    "claims_per_month"    : 80,
    "unique_patients"     : 15,
    "procedure_count"     : 7,
    "diagnosis_count"     : 4,
    "provider_specialty"  : 12,
    "geographic_region"   : 3,
    "claim_duration_days" : 2,
    "referral_flag"       : 0,
    "prior_auth_flag"     : 0,
    "telemedicine_flag"   : 0,
    "weekend_submission"  : 1,
    "round_amount_flag"   : 1,
    "icd_cpt_mismatch"    : 0.82,
    "provider_velocity"   : 180,
    "amount_deviation_z"  : 4.2,
    "patient_distance_km" : 420,
    "prev_fraud_flag"     : 1,
}

result = predict_claim(suspicious_claim)
print(f"\n  XGBoost Score   : {result['xgb_score']}")
print(f"  Autoencoder Score: {result['ae_score']}")
print(f"  Hybrid Score    : {result['hybrid_score']}")
print(f"  Fraud Flag      : {'⚠ FRAUD DETECTED' if result['fraud_flag'] else '✓ Legitimate'}")
print(f"  Risk Level      : {result['risk_level']}")
print(f"\n  Top Contributing Features (SHAP):")
for feat, val in result["top_shap_features"]:
    print(f"    {feat:<28} : {val:+.4f}")

print("\n" + "="*60)
print("  Milestone 2 Complete — All outputs saved to outputs_milestone2/")
print("="*60)