
# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os, time, warnings, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import (train_test_split, StratifiedKFold,
                                         cross_val_score, learning_curve)
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (classification_report, confusion_matrix,
                                         roc_auc_score, f1_score, precision_score,
                                         recall_score, matthews_corrcoef,
                                         average_precision_score, roc_curve,
                                         precision_recall_curve)
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, IsolationForest
from sklearn.svm                import SVC
from imblearn.over_sampling     import ADASYN

import xgboost as xgb
import shap

import tensorflow as tf
from tensorflow.keras.models     import Model
from tensorflow.keras.layers     import (Input, Dense, Dropout,
                                          BatchNormalization, LeakyReLU)
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    # Dataset
    "n_samples"          : 150_000,
    "fraud_ratio"        : 0.04,
    "test_size"          : 0.20,
    "val_size"           : 0.10,
    "random_state"       : 42,

    # Baseline XGBoost (Milestone 2 defaults)
    "xgb_n_estimators"   : 500,
    "xgb_max_depth"      : 6,
    "xgb_learning_rate"  : 0.05,

    # Autoencoder (Milestone 2 defaults)
    "ae_encoding_dim"    : 16,
    "ae_epochs"          : 60,
    "ae_batch_size"      : 256,
    "ae_threshold_pct"   : 95,

    # Hybrid scoring
    "hybrid_alpha"       : 0.65,
    "hybrid_beta"        : 0.35,

    # Cross-validation
    "cv_folds"           : 5,

    # Hyperparameter grid (XGBoost)
    "hp_grid" : {
        "n_estimators"  : [200, 500, 800],
        "max_depth"     : [4, 6, 8],
        "learning_rate" : [0.01, 0.05, 0.10],
        "subsample"     : [0.75, 0.85],
    },

    # Stress test
    "stress_batch_sizes" : [100, 500, 1_000, 5_000, 10_000],

    # Concept drift: inject new fraud pattern after X% of data
    "drift_split"        : 0.70,

    "output_dir"         : "outputs_milestone4",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

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

# ─────────────────────────────────────────────
# 2. DATA GENERATION (reused from Milestone 2)
# ─────────────────────────────────────────────
def generate_claims_dataset(n: int, fraud_ratio: float,
                             dataset_id: str,
                             drift: bool = False) -> pd.DataFrame:
    """
    Generate synthetic health insurance claims.
    drift=True injects a NEW fraud typology (round-trip billing scheme)
    not present in the original training distribution — used for
    concept drift simulation in Section 8.
    """
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    def _legit(size):
        return {
            "claim_amount"       : rng.lognormal(6.5, 1.1, size).clip(50, 80_000),
            "claim_quantity"     : rng.integers(1, 6,   size=size).astype(float),
            "provider_age"       : rng.integers(28, 70, size=size).astype(float),
            "patient_age"        : rng.integers(18, 90, size=size).astype(float),
            "days_since_last"    : rng.integers(1, 365, size=size).astype(float),
            "claims_per_month"   : rng.poisson(lam=2.1, size=size).clip(0,20).astype(float),
            "unique_patients"    : rng.integers(10, 500, size=size).astype(float),
            "procedure_count"    : rng.integers(1, 8,   size=size).astype(float),
            "diagnosis_count"    : rng.integers(1, 5,   size=size).astype(float),
            "provider_specialty" : rng.integers(0, 20,  size=size).astype(float),
            "geographic_region"  : rng.integers(0, 8,   size=size).astype(float),
            "claim_duration_days": rng.integers(1, 30,  size=size).astype(float),
            "referral_flag"      : rng.integers(0, 2,   size=size).astype(float),
            "prior_auth_flag"    : rng.integers(0, 2,   size=size).astype(float),
            "telemedicine_flag"  : rng.integers(0, 2,   size=size).astype(float),
            "weekend_submission" : rng.integers(0, 2,   size=size).astype(float),
            "round_amount_flag"  : rng.integers(0, 2,   size=size).astype(float),
            "icd_cpt_mismatch"  : rng.uniform(0, 0.15, size=size),
            "provider_velocity"  : rng.lognormal(3.0, 0.8, size=size).clip(1, 200),
            "amount_deviation_z" : rng.normal(0, 1,    size=size).clip(-3, 3),
            "patient_distance_km": rng.lognormal(2.5, 1.0, size=size).clip(0, 500),
            "prev_fraud_flag"    : rng.choice([0,1], size=size, p=[0.98,0.02]).astype(float),
        }

    def _fraud(size, drifted=False):
        d = _legit(size)
        if not drifted:
            # Original fraud typology (Milestone 2)
            d["claim_amount"]       *= rng.uniform(1.5, 4.0, size=size)
            d["icd_cpt_mismatch"]   = rng.uniform(0.4, 1.0, size=size)
            d["provider_velocity"]  *= rng.uniform(3, 10, size=size)
            d["amount_deviation_z"] = rng.normal(3.5, 1.2, size=size)
            d["round_amount_flag"]  = rng.choice([0,1], size=size, p=[0.3,0.7]).astype(float)
            d["prev_fraud_flag"]    = rng.choice([0,1], size=size, p=[0.6,0.4]).astype(float)
            d["days_since_last"]    = rng.integers(1, 15, size=size).astype(float)
        else:
            # NEW drift typology: round-trip billing (low amounts, high frequency,
            # geographically dispersed) — deliberately different pattern
            d["claim_amount"]       = rng.uniform(100, 500, size=size)  # low amounts
            d["claims_per_month"]   = rng.poisson(lam=45, size=size).clip(30,200).astype(float)
            d["patient_distance_km"]= rng.lognormal(5.5, 1.0, size=size).clip(200, 3000)
            d["icd_cpt_mismatch"]   = rng.uniform(0.05, 0.25, size=size)  # low mismatch!
            d["amount_deviation_z"] = rng.normal(-1.2, 0.8, size=size)    # BELOW average
            d["round_amount_flag"]  = rng.choice([0,1], size=size, p=[0.8,0.2]).astype(float)
            d["provider_velocity"]  *= rng.uniform(4, 12, size=size)
        return d

    legit_d  = _legit(n_legit)
    fraud_d  = _fraud(n_fraud, drifted=drift)
    rows = []
    for d, label in [(legit_d, 0), (fraud_d, 1)]:
        sub = pd.DataFrame(d)
        sub["label"]          = label
        sub["dataset_source"] = dataset_id
        rows.append(sub)
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=42)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_claim_amount"]      = np.log1p(df["claim_amount"])
    df["log_provider_velocity"] = np.log1p(df["provider_velocity"])
    df["amount_x_velocity"]     = df["amount_deviation_z"] * df["log_provider_velocity"]
    df["mismatch_x_velocity"]   = df["icd_cpt_mismatch"]   * df["log_provider_velocity"]
    df["claims_per_patient"]    = (df["claims_per_month"] /
                                   df["unique_patients"].clip(1)).clip(0, 50)
    df["high_risk_composite"]   = (
        df["prev_fraud_flag"] + df["round_amount_flag"] +
        df["weekend_submission"] + (df["icd_cpt_mismatch"] > 0.3).astype(int)
    )
    return df


def reconstruction_error(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred), axis=1)


def build_autoencoder(input_dim: int, encoding_dim: int = 16):
    inputs  = Input(shape=(input_dim,))
    x = Dense(64, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x); x = LeakyReLU(0.1)(x); x = Dropout(0.2)(x)
    x = Dense(32, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x); x = LeakyReLU(0.1)(x)
    encoded = Dense(encoding_dim, activation="relu", name="bottleneck")(x)
    x = Dense(32, kernel_regularizer=l2(1e-4))(encoded)
    x = BatchNormalization()(x); x = LeakyReLU(0.1)(x); x = Dropout(0.2)(x)
    x = Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x); x = LeakyReLU(0.1)(x)
    decoded = Dense(input_dim, activation="linear")(x)
    return Model(inputs, decoded, name="autoencoder")


def compute_hybrid(xgb_proba, ae_norm, alpha=0.65, beta=0.35):
    return alpha * xgb_proba + beta * ae_norm


def metrics_dict(y_true, y_pred, y_score) -> dict:
    return {
        "precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0),    4),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0),        4),
        "auc_roc"   : round(roc_auc_score(y_true, y_score),                   4),
        "avg_prec"  : round(average_precision_score(y_true, y_score),         4),
        "mcc"       : round(matthews_corrcoef(y_true, y_pred),                 4),
        "fpr"       : round((y_pred[y_true==0].sum() /
                             max((y_true==0).sum(), 1)),                       4),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
print("=" * 65)
print(" EmpowerTech Solutions — Milestone 4: Test & Optimize")
print("=" * 65)

# ─────────────────────────────────────────────
# 3. BUILD BASE DATASET
# ─────────────────────────────────────────────
print("\n[1/9] Building combined dataset...")
ds_a = generate_claims_dataset(80_000,  0.040, "CMS_Medicare")
ds_b = generate_claims_dataset(40_000,  0.035, "Private_Insurer")
ds_c = generate_claims_dataset(30_000,  0.050, "PMJAY_Government")
df_raw = pd.concat([ds_a, ds_b, ds_c], ignore_index=True)
df     = engineer_features(df_raw)

X = df[ALL_FEATURES].values
y = df["label"].values

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=CONFIG["test_size"], stratify=y,
    random_state=CONFIG["random_state"])
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=CONFIG["val_size"] / (1 - CONFIG["test_size"]),
    stratify=y_temp, random_state=CONFIG["random_state"])

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_val_sc    = scaler.transform(X_val)
X_test_sc   = scaler.transform(X_test)

adasyn      = ADASYN(sampling_strategy=0.25, random_state=42)
X_tr_res, y_tr_res = adasyn.fit_resample(X_train_sc, y_train)

print(f"    Dataset : {len(df):,} records | Fraud: {y.mean()*100:.2f}%")
print(f"    Train   : {len(X_tr_res):,} (after ADASYN) | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. TRAIN BASELINE HYBRID MODEL (Milestone 2)
# ─────────────────────────────────────────────
print("\n[2/9] Training Milestone 2 baseline hybrid model...")
spc = (y_tr_res==0).sum() / (y_tr_res==1).sum()

xgb_base = xgb.XGBClassifier(
    n_estimators    = CONFIG["xgb_n_estimators"],
    max_depth       = CONFIG["xgb_max_depth"],
    learning_rate   = CONFIG["xgb_learning_rate"],
    scale_pos_weight= spc,
    subsample       = 0.85,
    colsample_bytree= 0.80,
    min_child_weight= 5,
    gamma           = 0.1,
    reg_alpha       = 0.1,
    reg_lambda      = 1.0,
    use_label_encoder=False,
    eval_metric     = "aucpr",
    random_state    = 42,
    n_jobs          = -1,
    tree_method     = "hist",
)
xgb_base.fit(X_tr_res, y_tr_res,
             eval_set=[(X_val_sc, y_val)], verbose=False)

ae_base  = build_autoencoder(X_train_sc.shape[1], CONFIG["ae_encoding_dim"])
ae_base.compile(optimizer=Adam(1e-3), loss="mse")
X_legit  = X_train_sc[y_train == 0]
ae_base.fit(X_legit, X_legit,
            epochs=CONFIG["ae_epochs"],
            batch_size=CONFIG["ae_batch_size"],
            validation_split=0.10,
            callbacks=[EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
                       ReduceLROnPlateau(patience=4, factor=0.5, verbose=0)],
            verbose=0)

ae_train_err  = reconstruction_error(ae_base, X_legit)
ae_thresh     = np.percentile(ae_train_err, CONFIG["ae_threshold_pct"])
ae_max        = np.percentile(reconstruction_error(ae_base, X_test_sc), 99)

# Baseline hybrid test scores
xgb_test_p    = xgb_base.predict_proba(X_test_sc)[:,1]
ae_test_err   = reconstruction_error(ae_base, X_test_sc)
ae_test_norm  = np.clip(ae_test_err / ae_max, 0, 1)
hyb_test_sc   = compute_hybrid(xgb_test_p, ae_test_norm)

# Calibrate threshold on val set
xgb_val_p     = xgb_base.predict_proba(X_val_sc)[:,1]
ae_val_norm   = np.clip(reconstruction_error(ae_base, X_val_sc) / ae_max, 0, 1)
hyb_val_sc    = compute_hybrid(xgb_val_p, ae_val_norm)
prec_arr, rec_arr, thr_arr = precision_recall_curve(y_val, hyb_val_sc)
f1_arr   = 2*prec_arr*rec_arr / (prec_arr+rec_arr+1e-8)
best_thr = thr_arr[np.argmax(f1_arr[:-1])]

hyb_pred = (hyb_test_sc >= best_thr).astype(int)
base_m   = metrics_dict(y_test, hyb_pred, hyb_test_sc)
print(f"    Baseline Hybrid — F1: {base_m['f1']} | AUC: {base_m['auc_roc']} | "
      f"Threshold: {best_thr:.4f}")

# ─────────────────────────────────────────────
# 5. SECTION A: 5-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\n[3/9] Section A — 5-Fold Stratified Cross-Validation...")
skf     = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=42)
cv_f1s  = []
cv_aucs = []
cv_prec = []
cv_rec  = []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
    X_tr_f, X_vl_f = X[tr_idx], X[vl_idx]
    y_tr_f, y_vl_f = y[tr_idx], y[vl_idx]

    sc_f     = StandardScaler()
    X_tr_f   = sc_f.fit_transform(X_tr_f)
    X_vl_f   = sc_f.transform(X_vl_f)

    ada_f    = ADASYN(sampling_strategy=0.25, random_state=42)
    X_res_f, y_res_f = ada_f.fit_resample(X_tr_f, y_tr_f)

    spc_f    = (y_res_f==0).sum()/(y_res_f==1).sum()
    m        = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spc_f, subsample=0.85, colsample_bytree=0.80,
        use_label_encoder=False, eval_metric="aucpr",
        random_state=42, n_jobs=-1, tree_method="hist")
    m.fit(X_res_f, y_res_f, verbose=False)

    p_f = m.predict_proba(X_vl_f)[:,1]
    pr_f,rc_f,th_f = precision_recall_curve(y_vl_f, p_f)
    f1_f = 2*pr_f*rc_f/(pr_f+rc_f+1e-8)
    bt_f = th_f[np.argmax(f1_f[:-1])]
    pred_f = (p_f >= bt_f).astype(int)

    fold_f1  = f1_score(y_vl_f, pred_f, zero_division=0)
    fold_auc = roc_auc_score(y_vl_f, p_f)
    fold_pre = precision_score(y_vl_f, pred_f, zero_division=0)
    fold_rec = recall_score(y_vl_f, pred_f, zero_division=0)

    cv_f1s.append(fold_f1); cv_aucs.append(fold_auc)
    cv_prec.append(fold_pre); cv_rec.append(fold_rec)
    print(f"    Fold {fold}/5 — F1: {fold_f1:.4f} | AUC: {fold_auc:.4f} | "
          f"Precision: {fold_pre:.4f} | Recall: {fold_rec:.4f}")

print(f"\n    CV Summary — F1: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f} | "
      f"AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
cv_results = {
    "f1_mean": round(np.mean(cv_f1s),4), "f1_std": round(np.std(cv_f1s),4),
    "auc_mean":round(np.mean(cv_aucs),4),"auc_std":round(np.std(cv_aucs),4),
    "prec_mean":round(np.mean(cv_prec),4),"rec_mean":round(np.mean(cv_rec),4),
}

# ─────────────────────────────────────────────
# 6. SECTION B: HYPERPARAMETER OPTIMISATION
# ─────────────────────────────────────────────
print("\n[4/9] Section B — Hyperparameter Optimisation (Grid Search)...")
hp_grid   = CONFIG["hp_grid"]
hp_results= []
best_f1   = 0.0
best_params = {}

for n_est in hp_grid["n_estimators"]:
    for depth in hp_grid["max_depth"]:
        for lr in hp_grid["learning_rate"]:
            for sub in hp_grid["subsample"]:
                m_hp = xgb.XGBClassifier(
                    n_estimators=n_est, max_depth=depth,
                    learning_rate=lr, subsample=sub,
                    colsample_bytree=0.80, scale_pos_weight=spc,
                    use_label_encoder=False, eval_metric="aucpr",
                    random_state=42, n_jobs=-1, tree_method="hist")
                m_hp.fit(X_tr_res, y_tr_res, verbose=False)
                p_hp = m_hp.predict_proba(X_val_sc)[:,1]
                pr_hp,rc_hp,th_hp = precision_recall_curve(y_val, p_hp)
                f1_hp= 2*pr_hp*rc_hp/(pr_hp+rc_hp+1e-8)
                bt_hp= th_hp[np.argmax(f1_hp[:-1])]
                val_f1 = f1_score(y_val, (p_hp>=bt_hp).astype(int), zero_division=0)
                val_auc= roc_auc_score(y_val, p_hp)
                hp_results.append({
                    "n_estimators":n_est,"max_depth":depth,
                    "learning_rate":lr,"subsample":sub,
                    "val_f1":round(val_f1,4),"val_auc":round(val_auc,4)
                })
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_params = {"n_estimators":n_est,"max_depth":depth,
                                   "learning_rate":lr,"subsample":sub}

hp_df = pd.DataFrame(hp_results).sort_values("val_f1", ascending=False)
print(f"    Grid search: {len(hp_results)} combinations evaluated")
print(f"    Best params : {best_params}")
print(f"    Best val F1 : {best_f1:.4f}")
print(hp_df.head(5).to_string(index=False))

# ─────────────────────────────────────────────
# 7. SECTION C: OPTIMISED MODEL
# ─────────────────────────────────────────────
print("\n[5/9] Section C — Training Optimised Model with Best Hyperparameters...")
xgb_opt = xgb.XGBClassifier(
    **best_params,
    colsample_bytree = 0.80,
    min_child_weight = 5,
    gamma            = 0.1,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    scale_pos_weight = spc,
    use_label_encoder= False,
    eval_metric      = "aucpr",
    random_state     = 42,
    n_jobs           = -1,
    tree_method      = "hist",
)
xgb_opt.fit(X_tr_res, y_tr_res,
            eval_set=[(X_val_sc, y_val)], verbose=False)

# Optimised autoencoder: search over encoding dimensions
print("    Optimising Autoencoder encoding dimension...")
ae_enc_dims  = [8, 16, 32]
ae_best_loss = np.inf
ae_best_dim  = 16
for enc_dim in ae_enc_dims:
    ae_cand = build_autoencoder(X_train_sc.shape[1], enc_dim)
    ae_cand.compile(optimizer=Adam(1e-3), loss="mse")
    hist = ae_cand.fit(X_legit, X_legit,
                       epochs=30, batch_size=256,
                       validation_split=0.10, verbose=0,
                       callbacks=[EarlyStopping(patience=5,
                                  restore_best_weights=True, verbose=0)])
    val_loss = min(hist.history["val_loss"])
    print(f"      enc_dim={enc_dim:2d} → best val_loss={val_loss:.6f}")
    if val_loss < ae_best_loss:
        ae_best_loss = val_loss
        ae_best_dim  = enc_dim
        ae_opt       = ae_cand

print(f"    Best AE encoding dim: {ae_best_dim} (val_loss={ae_best_loss:.6f})")

# Optimised hybrid weights search
print("    Optimising hybrid weights (alpha/beta)...")
weight_results = []
for alpha in np.arange(0.50, 0.91, 0.05):
    beta = 1.0 - alpha
    ae_v_err  = reconstruction_error(ae_opt, X_val_sc)
    ae_v_max  = np.percentile(ae_v_err, 99)
    ae_v_norm = np.clip(ae_v_err / max(ae_v_max, 1e-9), 0, 1)
    xgb_v_p   = xgb_opt.predict_proba(X_val_sc)[:,1]
    h_sc      = alpha * xgb_v_p + beta * ae_v_norm
    pr_w,rc_w,th_w = precision_recall_curve(y_val, h_sc)
    f1_w = 2*pr_w*rc_w/(pr_w+rc_w+1e-8)
    bt_w = th_w[np.argmax(f1_w[:-1])]
    wf1  = f1_score(y_val, (h_sc>=bt_w).astype(int), zero_division=0)
    weight_results.append({"alpha":round(alpha,2),"beta":round(beta,2),"val_f1":round(wf1,4)})

w_df = pd.DataFrame(weight_results).sort_values("val_f1", ascending=False)
best_alpha = w_df.iloc[0]["alpha"]
best_beta  = w_df.iloc[0]["beta"]
print(f"    Best weights — alpha={best_alpha}, beta={best_beta}")
print(w_df.head(5).to_string(index=False))

# Final optimised hybrid test evaluation
ae_opt_max    = np.percentile(reconstruction_error(ae_opt, X_test_sc), 99)
xgb_opt_tp    = xgb_opt.predict_proba(X_test_sc)[:,1]
ae_opt_norm   = np.clip(reconstruction_error(ae_opt, X_test_sc) / ae_opt_max, 0, 1)
hyb_opt_sc    = best_alpha * xgb_opt_tp + best_beta * ae_opt_norm

xgb_opt_vp    = xgb_opt.predict_proba(X_val_sc)[:,1]
ae_opt_vn     = np.clip(reconstruction_error(ae_opt, X_val_sc) / ae_opt_max, 0, 1)
hyb_opt_vs    = best_alpha * xgb_opt_vp + best_beta * ae_opt_vn
pr_o,rc_o,th_o = precision_recall_curve(y_val, hyb_opt_vs)
f1_o = 2*pr_o*rc_o/(pr_o+rc_o+1e-8)
bt_o = th_o[np.argmax(f1_o[:-1])]

opt_pred  = (hyb_opt_sc >= bt_o).astype(int)
opt_m     = metrics_dict(y_test, opt_pred, hyb_opt_sc)
print(f"\n    Optimised Hybrid — F1: {opt_m['f1']} | AUC: {opt_m['auc_roc']} | "
      f"Threshold: {bt_o:.4f}")

# ─────────────────────────────────────────────
# 8. SECTION D: BASELINE MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n[6/9] Section D — Baseline Model Comparison...")

# Sample down for slower models (SVM, RF)
n_samp    = min(20_000, len(X_tr_res))
idx_samp  = np.random.choice(len(X_tr_res), n_samp, replace=False)
X_samp    = X_tr_res[idx_samp]
y_samp    = y_tr_res[idx_samp]

baselines = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1),
    "XGBoost (baseline M2)": xgb_base,
    "XGBoost (optimised)":   xgb_opt,
}

baseline_results = {}
print(f"    {'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8} {'MCC':>8} {'FPR':>8}")
print("    " + "-"*82)
for name, model in baselines.items():
    if name not in ("XGBoost (baseline M2)", "XGBoost (optimised)"):
        model.fit(X_samp, y_samp)
    p   = model.predict_proba(X_test_sc)[:,1]
    pr_b,rc_b,th_b = precision_recall_curve(y_test, p)
    f1_b = 2*pr_b*rc_b/(pr_b+rc_b+1e-8)
    bt_b = th_b[np.argmax(f1_b[:-1])]
    m    = metrics_dict(y_test, (p>=bt_b).astype(int), p)
    baseline_results[name] = m
    print(f"    {name:<30} {m['precision']:>10} {m['recall']:>10} "
          f"{m['f1']:>8} {m['auc_roc']:>8} {m['mcc']:>8} {m['fpr']:>8}")

# Isolation Forest (unsupervised)
iso = IsolationForest(contamination=0.04, random_state=42, n_jobs=-1)
iso.fit(X_tr_res[y_tr_res==0])   # train on legitimate only
iso_scores = -iso.score_samples(X_test_sc)
iso_norm   = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
pr_i,rc_i,th_i = precision_recall_curve(y_test, iso_norm)
f1_i = 2*pr_i*rc_i/(pr_i+rc_i+1e-8)
bt_i = th_i[np.argmax(f1_i[:-1])]
iso_m = metrics_dict(y_test, (iso_norm>=bt_i).astype(int), iso_norm)
baseline_results["Isolation Forest"] = iso_m
print(f"    {'Isolation Forest':<30} {iso_m['precision']:>10} {iso_m['recall']:>10} "
      f"{iso_m['f1']:>8} {iso_m['auc_roc']:>8} {iso_m['mcc']:>8} {iso_m['fpr']:>8}")

# Optimised Hybrid (final)
baseline_results["Hybrid Optimised (M4)"] = opt_m
print(f"    {'Hybrid Optimised (M4)':<30} {opt_m['precision']:>10} {opt_m['recall']:>10} "
      f"{opt_m['f1']:>8} {opt_m['auc_roc']:>8} {opt_m['mcc']:>8} {opt_m['fpr']:>8}")

# ─────────────────────────────────────────────
# 9. SECTION E: CONCEPT DRIFT SIMULATION
# ─────────────────────────────────────────────
print("\n[7/9] Section E — Concept Drift Simulation...")

# Build drift dataset: first 70% = original patterns, last 30% = new drift pattern
n_drift   = int(CONFIG["n_samples"] * (1-CONFIG["drift_split"]))
df_drift  = generate_claims_dataset(n_drift, 0.05, "DRIFT", drift=True)
df_drift  = engineer_features(df_drift)
X_drift   = scaler.transform(df_drift[ALL_FEATURES].values)
y_drift   = df_drift["label"].values

# Evaluate baseline model on drift data
xgb_drift_p  = xgb_base.predict_proba(X_drift)[:,1]
ae_drift_n   = np.clip(reconstruction_error(ae_base, X_drift) / ae_max, 0, 1)
hyb_drift_sc = compute_hybrid(xgb_drift_p, ae_drift_n)
drift_pred   = (hyb_drift_sc >= best_thr).astype(int)
drift_base_m = metrics_dict(y_drift, drift_pred, hyb_drift_sc)

# Evaluate optimised model on drift data
xgb_d_opt_p  = xgb_opt.predict_proba(X_drift)[:,1]
ae_d_opt_n   = np.clip(reconstruction_error(ae_opt, X_drift) / ae_opt_max, 0, 1)
hyb_d_opt_sc = best_alpha*xgb_d_opt_p + best_beta*ae_d_opt_n
drift_opt_pr,drift_opt_rc,drift_opt_th = precision_recall_curve(y_drift, hyb_d_opt_sc)
drift_f1_arr = 2*drift_opt_pr*drift_opt_rc/(drift_opt_pr+drift_opt_rc+1e-8)
drift_bt     = drift_opt_th[np.argmax(drift_f1_arr[:-1])]
drift_opt_m  = metrics_dict(y_drift, (hyb_d_opt_sc>=drift_bt).astype(int), hyb_d_opt_sc)

print(f"    Drift dataset: {len(df_drift):,} records | Fraud: {y_drift.mean()*100:.2f}%")
print(f"    Baseline M2 on drift data  — F1: {drift_base_m['f1']} | AUC: {drift_base_m['auc_roc']}")
print(f"    Optimised M4 on drift data — F1: {drift_opt_m['f1']} | AUC: {drift_opt_m['auc_roc']}")
print(f"    F1 degradation (M2 baseline): "
      f"{((base_m['f1']-drift_base_m['f1'])/base_m['f1']*100):.1f}%")
print(f"    F1 degradation (M4 optimised): "
      f"{((opt_m['f1']-drift_opt_m['f1'])/opt_m['f1']*100):.1f}%")

# ─────────────────────────────────────────────
# 10. SECTION F: STRESS TEST (THROUGHPUT)
# ─────────────────────────────────────────────
print("\n[8/9] Section F — Stress Test: Throughput Benchmarking...")
stress_results = []
for batch_size in CONFIG["stress_batch_sizes"]:
    idx_s = np.random.choice(len(X_test_sc), batch_size, replace=False)
    Xb    = X_test_sc[idx_s]
    t0    = time.perf_counter()
    xgb_b_p   = xgb_opt.predict_proba(Xb)[:,1]
    ae_b_err  = reconstruction_error(ae_opt, Xb)
    ae_b_norm = np.clip(ae_b_err / ae_opt_max, 0, 1)
    hyb_b     = best_alpha * xgb_b_p + best_beta * ae_b_norm
    _         = (hyb_b >= bt_o).astype(int)
    elapsed   = time.perf_counter() - t0
    ms_per    = (elapsed / batch_size) * 1_000
    throughput= batch_size / elapsed
    stress_results.append({
        "batch_size": batch_size,
        "total_ms"  : round(elapsed*1000, 1),
        "ms_per_claim": round(ms_per, 3),
        "claims_per_sec": round(throughput, 0),
    })
    print(f"    Batch {batch_size:>6,} claims → {elapsed*1000:>7.1f}ms total | "
          f"{ms_per:.3f}ms/claim | {throughput:>8,.0f} claims/sec")

# ─────────────────────────────────────────────
# 11. SECTION G: SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n[9/9] Section G — SHAP Feature Importance (Optimised Model)...")
try:
    booster = xgb_opt.get_booster()
    booster.set_param({"base_score": 0.5})
    explainer   = shap.TreeExplainer(booster)
    shap_sample = X_test_sc[:500]
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values)>1 else shap_values[0]
    if hasattr(shap_values,"ndim") and shap_values.ndim == 2:
        shap_mean = np.abs(shap_values).mean(axis=0)
    else:
        shap_mean = np.abs(shap_values)
    shap_df = pd.DataFrame({"Feature":ALL_FEATURES,"SHAP_mean":shap_mean})\
                .sort_values("SHAP_mean", ascending=False).head(15)
    print("    Top 10 SHAP features (Optimised Model):")
    for _, row in shap_df.head(10).iterrows():
        bar = "█" * int(row["SHAP_mean"] / shap_df["SHAP_mean"].max() * 30)
        print(f"      {row['Feature']:<28} {bar} {row['SHAP_mean']:.4f}")
    shap_ok = True
except Exception as e:
    print(f"    SHAP skipped: {e}")
    shap_df  = pd.DataFrame({"Feature":ALL_FEATURES,"SHAP_mean":np.zeros(len(ALL_FEATURES))})
    shap_ok  = False

# ─────────────────────────────────────────────
# 12. GENERATE PLOTS
# ─────────────────────────────────────────────
print("\n  Generating plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("EmpowerTech Solutions — Milestone 4: Test & Optimize Algorithm",
             fontsize=14, fontweight="bold", y=1.01)

# ── Plot 1: CV Fold performance ──
ax = axes[0,0]
folds  = [f"Fold {i+1}" for i in range(CONFIG["cv_folds"])]
x_pos  = np.arange(len(folds))
bars   = ax.bar(x_pos-0.2, cv_f1s,  0.35, label="F1-Score",  color="#2563EB", alpha=0.85)
bars2  = ax.bar(x_pos+0.2, cv_aucs, 0.35, label="AUC-ROC",   color="#10B981", alpha=0.85)
ax.axhline(np.mean(cv_f1s),  color="#2563EB", linestyle="--", lw=1.2,
           label=f"Mean F1  = {np.mean(cv_f1s):.4f}")
ax.axhline(np.mean(cv_aucs), color="#10B981", linestyle="--", lw=1.2,
           label=f"Mean AUC = {np.mean(cv_aucs):.4f}")
ax.set_xticks(x_pos); ax.set_xticklabels(folds); ax.set_ylim(0.95, 1.02)
ax.set_title("5-Fold Cross-Validation", fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ── Plot 2: Hyperparameter sensitivity (top 10) ──
ax = axes[0,1]
top10 = hp_df.head(10)
ax.barh(range(len(top10)), top10["val_f1"], color="#6D28D9", alpha=0.8)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(
    [f"n={r['n_estimators']} d={r['max_depth']} lr={r['learning_rate']}"
     for _,r in top10.iterrows()], fontsize=8)
ax.set_xlabel("Validation F1-Score")
ax.set_title("Top 10 Hyperparameter Combinations", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ── Plot 3: ROC Curves — all models ──
ax = axes[0,2]
roc_models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=500,class_weight="balanced").fit(X_samp,y_samp),
        X_test_sc),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100,class_weight="balanced",
                               random_state=42,n_jobs=-1).fit(X_samp,y_samp),
        X_test_sc),
    "XGBoost (M2 Baseline)": (xgb_base, X_test_sc),
    "XGBoost (M4 Optimised)": (xgb_opt, X_test_sc),
}
cols_roc = ["#94A3B8","#F59E0B","#2563EB","#10B981"]
for (name, (model, Xeval)), col in zip(roc_models.items(), cols_roc):
    p_roc  = model.predict_proba(Xeval)[:,1]
    fpr_r,tpr_r,_ = roc_curve(y_test, p_roc)
    auc_r  = roc_auc_score(y_test, p_roc)
    lw = 2.5 if "M4" in name else 1.5
    ax.plot(fpr_r, tpr_r, lw=lw, color=col, label=f"{name} (AUC={auc_r:.4f})")

# Add hybrid optimised
fpr_h,tpr_h,_ = roc_curve(y_test, hyb_opt_sc)
auc_h = roc_auc_score(y_test, hyb_opt_sc)
ax.plot(fpr_h, tpr_h, lw=3, color="#EF4444",
        label=f"Hybrid Optimised M4 (AUC={auc_h:.4f})")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison — All Models", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ── Plot 4: Model comparison bar chart ──
ax = axes[1,0]
comp_names = list(baseline_results.keys())
comp_f1s   = [baseline_results[k]["f1"]      for k in comp_names]
comp_aucs  = [baseline_results[k]["auc_roc"] for k in comp_names]
x_c = np.arange(len(comp_names)); w_c = 0.35
ax.bar(x_c-w_c/2, comp_f1s,  w_c, label="F1-Score", color="#2563EB", alpha=0.85)
ax.bar(x_c+w_c/2, comp_aucs, w_c, label="AUC-ROC",  color="#10B981", alpha=0.85)
ax.set_xticks(x_c)
ax.set_xticklabels([n.replace(" ","\n").replace("(","\n(")
                    for n in comp_names], fontsize=7)
ax.set_ylim(0.5, 1.05)
ax.set_title("Model Comparison: F1 vs AUC-ROC", fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ── Plot 5: Throughput stress test ──
ax = axes[1,1]
st_df = pd.DataFrame(stress_results)
ax.plot(st_df["batch_size"], st_df["ms_per_claim"],
        marker="o", lw=2.5, color="#6D28D9", markersize=6)
ax.axhline(y=15, color="#EF4444", linestyle="--", lw=1.5,
           label="15ms real-time budget")
ax.set_xscale("log")
ax.set_xlabel("Batch Size (log scale)")
ax.set_ylabel("Latency per Claim (ms)")
ax.set_title("Stress Test: Latency vs Batch Size", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for _, row in st_df.iterrows():
    ax.annotate(f"{row['ms_per_claim']}ms",
                (row["batch_size"], row["ms_per_claim"]),
                textcoords="offset points", xytext=(0,8),
                ha="center", fontsize=8)

# ── Plot 6: SHAP top features ──
ax = axes[1,2]
if shap_ok:
    colors_s = ["#EF4444" if i<5 else "#F59E0B" if i<10 else "#2563EB"
                for i in range(len(shap_df))]
    ax.barh(range(len(shap_df)), shap_df["SHAP_mean"], color=colors_s, alpha=0.85)
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df["Feature"], fontsize=8.5)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top 15 Features — SHAP Importance\n(Optimised Model)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
else:
    xgb_imp = pd.Series(
        xgb_opt.get_booster().get_fscore()
    ).sort_values(ascending=False).head(15)
    ax.barh(range(len(xgb_imp)), xgb_imp.values, color="#2563EB", alpha=0.85)
    ax.set_yticks(range(len(xgb_imp)))
    ax.set_yticklabels(xgb_imp.index, fontsize=8.5)
    ax.set_xlabel("XGBoost Feature Importance (F-Score)")
    ax.set_title("Top 15 Features — XGBoost Importance\n(SHAP fallback)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = os.path.join(CONFIG["output_dir"], "milestone4_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 13. SAVE OPTIMISED MODELS
# ─────────────────────────────────────────────
models_dir = "outputs_milestone2"
os.makedirs(models_dir, exist_ok=True)
with open(os.path.join(models_dir, "xgb_model_optimised.pkl"), "wb") as f:
    pickle.dump(xgb_opt, f)
with open(os.path.join(models_dir, "scaler_optimised.pkl"), "wb") as f:
    pickle.dump(scaler, f)
ae_opt.save(os.path.join(models_dir, "autoencoder_optimised.keras"))

# ─────────────────────────────────────────────
# 14. FINAL SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MILESTONE 4 — COMPLETE RESULTS SUMMARY")
print("=" * 65)

print("\n  A. 5-Fold Cross-Validation (XGBoost, ADASYN):")
print(f"     F1  : {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
print(f"     AUC : {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}")

print("\n  B. Best Hyperparameters (Grid Search):")
for k, v in best_params.items():
    print(f"     {k:<20}: {v}")

print(f"\n  C. Optimised Model Performance (Test Set):")
print(f"     Precision : {opt_m['precision']}")
print(f"     Recall    : {opt_m['recall']}")
print(f"     F1-Score  : {opt_m['f1']}")
print(f"     AUC-ROC   : {opt_m['auc_roc']}")
print(f"     MCC       : {opt_m['mcc']}")
print(f"     FPR       : {opt_m['fpr']}")
print(f"     Best alpha: {best_alpha} | Best beta: {best_beta}")
print(f"     Threshold : {bt_o:.4f}")

print(f"\n  D. Baseline vs Optimised (Test F1):")
for name, m in baseline_results.items():
    tag = " ← Optimised" if "M4" in name else ""
    print(f"     {name:<30}: F1={m['f1']} | AUC={m['auc_roc']}{tag}")

print(f"\n  E. Concept Drift Simulation:")
print(f"     M2 Baseline on drift data  — F1: {drift_base_m['f1']} | AUC: {drift_base_m['auc_roc']}")
print(f"     M4 Optimised on drift data — F1: {drift_opt_m['f1']} | AUC: {drift_opt_m['auc_roc']}")
print(f"     M2 F1 degradation: "
      f"{abs(base_m['f1']-drift_base_m['f1'])/base_m['f1']*100:.1f}%")
print(f"     M4 F1 degradation: "
      f"{abs(opt_m['f1']-drift_opt_m['f1'])/opt_m['f1']*100:.1f}%")

print(f"\n  F. Stress Test (Optimised Model):")
for r in stress_results:
    print(f"     {r['batch_size']:>6,} claims → {r['ms_per_claim']} ms/claim | "
          f"{int(r['claims_per_sec']):,} claims/sec")

print(f"\n  Plots saved to '{CONFIG['output_dir']}/milestone4_results.png'")
print(f"  Models saved to '{models_dir}/'")
print("=" * 65)
print("  Milestone 4 Complete — Test and Optimize Algorithm")
print("=" * 65)