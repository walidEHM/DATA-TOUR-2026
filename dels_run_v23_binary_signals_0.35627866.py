import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("V23 — BINARY FRAUD SIGNALS +ARCHITECTUR DU V12")
print("=" * 60)

DATA_DIR = Path("dataset")
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

eps = 1e-6
df_tr_ref = train[train['period'] < 90].copy()
global_mean = df_tr_ref["fraud_flag"].mean()
K = 100

print("1. Features de base du V12 (20 features)...")
def build_features(df, df_ref):
    df = df.copy()
    #  Transactionnelles
    df["is_op03"] = (df["operation"] == "op_03").astype(int)
    df["amount_log"]  = np.log1p(df["amount"])
    df["amount_sqrt"] = np.sqrt(np.maximum(df["amount"], 0))
    df["in_fraud_range"] = ((df["amount"] >= 19000) & (df["amount"] <= 600000)).astype(int)
    df["op03_in_range"]  = (df["is_op03"] & df["in_fraud_range"]).astype(int)
    df["orig_chg"] = df["origin_balance_after"] - df["origin_balance_before"]
    df["amt_to_orig"] = df["amount"] / (df["origin_balance_before"].abs() + eps)
    df["amt_to_dest"] = df["amount"] / (df["destination_balance_before"].abs() + eps)
    df["orig_error_ratio"] = np.abs(df["orig_chg"] + df["amount"]) / (df["amount"] + eps)
    df["has_orig_error"] = (df["orig_error_ratio"] > 0.01).astype(int)
    df["orig_no_move"] = (np.abs(df["orig_chg"]) < 1.0).astype(int)
    
    # NOUVELLES: Signaux binaires ultra-spécifiques à la fraude mobile

    # 1. Compte origin VIDÉ après la transaction (drain total)
    df["orig_fully_drained"]  = (df["origin_balance_after"] < 1.0).astype(int)

    # 2. Compte origin étai non-vide et s'est retrouvé vid (drain complet)
    df["orig_drained_from_nonzero"] = (
        (df["origin_balance_before"] > 1000) & 
        (df["origin_balance_after"] < 1.0)
    ).astype(int)

    # 3 Compte destination était VIDE avant de recevoir (compte fantôme)
    df["dest_was_empty"]      = (df["destination_balance_before"] < 1.0).astype(int)

    # 4. Destination vide + op03 + motant élevé = shéma fraude classique
    df["dest_empty_op03"]     = (df["dest_was_empty"] & df["is_op03"]).astype(int)

    # 5 Le changement du solde destination ne correspond PAS au montant (anomalie destination)
    df["dest_chg"] = df["destination_balance_after"] - df["destination_balance_before"]
    df["dest_error_ratio"] = np.abs(df["dest_chg"] - df["amount"]) / (df["amount"] + eps)
    df["has_dest_error"] = (df["dest_error_ratio"] > 0.01).astype(int)

    # 6. Aucun mouvement côté destination (fraude silencieuse)
    df["dest_no_move"] = (np.abs(df["dest_chg"]) < 1.0).astype(int)

    # 7. Transfert integral du solde du compte d'origine
    df["orig_pct_sent"] = df["amount"] / (df["origin_balance_before"] + eps)
    df["full_balance_sent"] = (df["orig_pct_sent"] > 0.99).astype(int)
    
    # Target Encoding (origin_account, operation)
    for col in ["origin_account", "operation"]:
        stats    = df_ref.groupby(col)["fraud_flag"].agg(["sum", "count"])
        smoothed = (stats["sum"] + K * global_mean) / (stats["count"] + K)
        df[f"{col}_te"] = df[col].map(smoothed).fillna(global_mean)
    freq_orig = df_ref["origin_account"].value_counts(normalize=True)
    df["origin_account_freq"] = df["origin_account"].map(freq_orig).fillna(0)

    orig_agg = df_ref.groupby("origin_account").agg(
        orig_n_tx=("fraud_flag", "count"), orig_n_dest=("destination_account", "nunique"),
        orig_amount_max=("amount", "max")).fillna(0)
    dest_agg = df_ref.groupby("destination_account").agg(
        dest_amount_mean=("amount", "mean"), dest_amount_std=("amount", "std")).fillna(0)
    df = df.join(orig_agg, on="origin_account")
    df = df.join(dest_agg, on="destination_account")
    for c in list(orig_agg.columns) + list(dest_agg.columns):
        if c in df.columns:
            df[c] = df[c].fillna(0)
    df["amt_vs_max"] = df["amount"] / (df["orig_amount_max"] + eps)
    return df

train_fe = build_features(train, df_tr_ref)
test_fe  = build_features(test,  df_tr_ref)

FEATURES = [
    # V12 base (20)
    "op03_in_range", "origin_account_te", "operation_te",
    "orig_amount_max", "is_op03", "amount_log", "amount", "amount_sqrt",
    "dest_amount_mean", "orig_n_dest", "amt_to_orig", "amt_to_dest",
    "origin_account_freq", "orig_n_tx", "dest_amount_std", "orig_chg",
    "orig_error_ratio", "has_orig_error", "orig_no_move", "amt_vs_max",
    # Signaux binaires fraude (7 nouveaux)
    "orig_fully_drained", "orig_drained_from_nonzero", "dest_was_empty",
    "dest_empty_op03", "has_dest_error", "dest_no_move", "full_balance_sent",
]
print(f"   {len(FEATURES)} features ({len(FEATURES)-20} nouvelles).")

df_tr  = train_fe[train_fe['period'] < 90]
df_val = train_fe[train_fe['period'] >= 90]
X_tr,   y_tr   = df_tr[FEATURES].fillna(0),    df_tr['fraud_flag']
X_val,  y_val  = df_val[FEATURES].fillna(0),   df_val['fraud_flag']
X_full, y_full = train_fe[FEATURES].fillna(0), train_fe['fraud_flag']
X_test         = test_fe[FEATURES].fillna(0)

scale_pos = float((y_tr == 0).sum() / (y_tr == 1).sum())

# validation locale rapide avec la graine 42 pour avoir une idée avant l'entraînement complet
print("\n2. Validation locale rapide (seed 42)...")

lgb_quick = lgb.LGBMClassifier(
    objective="binary", metric="average_precision", learning_rate=0.03,
    num_leaves=63, max_depth=-1, min_child_samples=100,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
    lambda_l1=1.0, lambda_l2=1.0, min_gain_to_split=0.01,
    scale_pos_weight=scale_pos, n_estimators=3000, random_state=42, verbose=-1, n_jobs=-1
)

lgb_quick.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(200, verbose=False)])
val_preds = lgb_quick.predict_proba(X_val)[:, 1]
val_prauc = average_precision_score(y_val, val_preds)
print(f"   Réussi : Val PR-AUC (seed 42): {val_prauc:.5f}  |  best_iter={lgb_quick.best_iteration_}")

SEEDS = [42, 123, 777, 2024, 8888, 1337, 9999]
lgb_preds, xgb_preds = [], []

print(f"\n3. Seed Averaging sur {len(SEEDS)} graines...")
for i, seed in enumerate(SEEDS):
    print(f"\n  --- Seed {i+1}/{len(SEEDS)} : {seed} ---")
    lgb_params = dict(
        objective="binary", metric="average_precision", learning_rate=0.03,
        num_leaves=63, max_depth=-1, min_child_samples=100,
        feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
        lambda_l1=1.0, lambda_l2=1.0, min_gain_to_split=0.01,
        scale_pos_weight=scale_pos, n_estimators=3000,
        random_state=seed, verbose=-1, n_jobs=-1
    )
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(200, verbose=False)])
    opt = max(100, int(m.best_iteration_ * (105 / 90)))
    print(f"    LGB: best_iter={m.best_iteration_}, final={opt}")
    fin_lgb = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": opt, "metric": None})
    fin_lgb.fit(X_full, y_full)
    lgb_preds.append(fin_lgb.predict_proba(X_test)[:, 1])

    m_xgb = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        learning_rate=0.03, max_depth=6, subsample=0.7, colsample_bytree=0.7,
        scale_pos_weight=scale_pos, n_estimators=3000,
        random_state=seed, n_jobs=-1, early_stopping_rounds=200
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    opt_xgb = max(100, int(m_xgb.best_iteration * (105 / 90)))
    print(f"    XGB: best_iter={m_xgb.best_iteration}, final={opt_xgb}")
    fin_xgb = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.03,
        max_depth=6, subsample=0.7, colsample_bytree=0.7,
        scale_pos_weight=scale_pos, n_estimators=opt_xgb, random_state=seed, n_jobs=-1
    )
    fin_xgb.fit(X_full, y_full)
    xgb_preds.append(fin_xgb.predict_proba(X_test)[:, 1])

print("\n4. Rank Averaging (60% LGB / 40% XGB)...")
rank_lgb = np.mean([pd.Series(p).rank() / len(p) for p in lgb_preds], axis=0)
rank_xgb = np.mean([pd.Series(p).rank() / len(p) for p in xgb_preds], axis=0)
final    = (rank_lgb * 0.60) + (rank_xgb * 0.40)

sub = pd.DataFrame({'id': test['id'], 'target': final})
out_file = "dataset/submission_v_BINARY_SIGNALS.csv"
sub.to_csv(out_file, index=False)
print(f"\nFait : {out_file}")
