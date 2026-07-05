import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("Chargement des données...")
DATA_DIR = Path("dataset")
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

def build_transactional_features(df):
    df = df.copy()
    eps = 1e-6
    df["is_op03"] = (df["operation"] == "op_03").astype(int)
    df["amount_log"]  = np.log1p(df["amount"])
    df["amount_sqrt"] = np.sqrt(np.maximum(df["amount"], 0))
    df["in_fraud_range"] = ((df["amount"] >= 19000) & (df["amount"] <= 600000)).astype(int)
    df["op03_in_range"]  = (df["is_op03"] & df["in_fraud_range"]).astype(int)
    df["orig_chg"] = df["origin_balance_after"] - df["origin_balance_before"]
    df["amt_to_orig"] = df["amount"] / (df["origin_balance_before"].abs() + eps)
    df["amt_to_dest"] = df["amount"] / (df["destination_balance_before"].abs() + eps)
    
    # NOUVELLES FEATURES ULTRA ROBUSTES DE LA V9 (pa de risque de leakage)
    df["orig_error_ratio"] = np.abs(df["orig_chg"] + df["amount"]) / (df["amount"] + eps)
    df["has_orig_error"] = (df["orig_error_ratio"] > 0.01).astype(int)
    df["orig_no_move"] = (np.abs(df["orig_chg"]) < 1.0).astype(int)
    
    return df

def build_account_features(df_target, df_source):
    df = df_target.copy()
    global_mean = df_source["fraud_flag"].mean()
    K = 100 
    eps = 1e-6
    for col in ["origin_account", "operation"]:
        stats = df_source.groupby(col)["fraud_flag"].agg(["sum", "count"])
        smoothed = (stats["sum"] + K * global_mean) / (stats["count"] + K)
        df[f"{col}_te"] = df[col].map(smoothed).fillna(global_mean)

    for col in ["origin_account"]:
        freq = df_source[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0)

    orig_agg = df_source.groupby("origin_account").agg(
        orig_n_tx=("fraud_flag", "count"),
        orig_n_dest=("destination_account", "nunique"),
        orig_amount_max=("amount", "max"),
    ).fillna(0)

    dest_agg = df_source.groupby("destination_account").agg(
        dest_amount_mean=("amount", "mean"),
        dest_amount_std=("amount", "std"),
    ).fillna(0)

    df = df.join(orig_agg, on="origin_account")
    df = df.join(dest_agg, on="destination_account")

    agg_cols = list(orig_agg.columns) + list(dest_agg.columns)
    for c in agg_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df_source[c].median() if c in df_source.columns else 0)
            
    # rATIO COMPORTEMENTAL
    df["amt_vs_max"] = df["amount"] / (df["orig_amount_max"] + eps)
    
    return df

# Prrparation des splits de donnees
df_tr  = train[train.period < 90].copy()
df_val = train[train.period >= 90].copy()

print("Ingénierie des features en cours...")
df_tr  = build_transactional_features(df_tr)
df_val = build_transactional_features(df_val)
test_tr = build_transactional_features(test)
train_tr = build_transactional_features(train)

df_val = build_account_features(df_val, df_source=df_tr)
df_tr  = build_account_features(df_tr,  df_source=df_tr)

train_full = build_account_features(train_tr, df_source=train_tr)
test_full  = build_account_features(test_tr,  df_source=train_tr)

FEATURES = [
    "op03_in_range", "origin_account_te", "operation_te",
    "orig_amount_max", "is_op03", "amount_log", "amount", "amount_sqrt",
    "dest_amount_mean", "orig_n_dest", "amt_to_orig", "amt_to_dest",
    "origin_account_freq", "orig_n_tx", "dest_amount_std", "orig_chg",
    # Nouvelles features injectées
    "orig_error_ratio", "has_orig_error", "orig_no_move", "amt_vs_max"
]

X_tr, y_tr = df_tr[FEATURES].fillna(0), df_tr["fraud_flag"]
X_val, y_val = df_val[FEATURES].fillna(0), df_val["fraud_flag"]

X_full, y_full = train_full[FEATURES].fillna(0), train_full["fraud_flag"]
X_test_full = test_full[FEATURES].fillna(0)

scale_pos = float((y_tr == 0).sum() / (y_tr == 1).sum())

# Definition des graine pour stabilise la variance
SEEDS = [42, 123, 777, 2024, 8888]

lgb_predictions = []
xgb_predictions = []

print("\n" + "="*50)
print(f"Lancement du Seed Averaging v12 sur {len(SEEDS)} graines")
print("="*50)

for i, seed in enumerate(SEEDS):
    print(f"\n--- Iteration {i+1}/{len(SEEDS)} | Graine : {seed} ---")
    
    # ------------------ LIGHTGBM ------------------
    lgb_params = dict(
        objective="binary", metric="average_precision", learning_rate=0.03,
        num_leaves=63, max_depth=-1, min_child_samples=100,
        feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
        lambda_l1=1.0, lambda_l2=1.0, min_gain_to_split=0.01,
        scale_pos_weight=scale_pos,
        n_estimators=3000, random_state=seed, verbose=-1, n_jobs=-1
    )
    
    print("  > LightGBM (Recherche d'arbres...)")
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150, verbose=False)])
    
    optimal_trees_lgb = int(model_lgb.best_iteration_ * (105 / 90))
    
    print(f"  > LightGBM (Entraînement Final - {optimal_trees_lgb} arbres...)")
    final_lgb_params = {**lgb_params}
    final_lgb_params["n_estimators"] = optimal_trees_lgb
    del final_lgb_params["metric"]
    
    final_lgb = lgb.LGBMClassifier(**final_lgb_params)
    final_lgb.fit(X_full, y_full)
    lgb_predictions.append(final_lgb.predict_proba(X_test_full)[:, 1])
    
    # ------------------ XGBOOST ------------------
    xgb_params = dict(
        objective="binary:logistic", eval_metric="aucpr", learning_rate=0.03,
        max_depth=6, subsample=0.7, colsample_bytree=0.7,
        scale_pos_weight=scale_pos,
        n_estimators=3000, random_state=seed, n_jobs=-1,
        early_stopping_rounds=150
    )
    
    print("  > XGBoost (Recherche d'arbres...)")
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    optimal_trees_xgb = int(model_xgb.best_iteration * (105 / 90))
    
    print(f"  > XGBoost (Entraînement Final - {optimal_trees_xgb} arbres...)")
    final_xgb_params = {**xgb_params}
    final_xgb_params["n_estimators"] = optimal_trees_xgb
    final_xgb_params["early_stopping_rounds"] = None
    
    final_xgb = xgb.XGBClassifier(**final_xgb_params)
    final_xgb.fit(X_full, y_full)
    xgb_predictions.append(final_xgb.predict_proba(X_test_full)[:, 1])

print("\n" + "="*50)
print("Fusion mathématique finale (Rank Averaging des 10 modèles)")

# Calcul du rang moyen pour tou les LGBM et tous les XGBoost
rank_lgb_avg = np.mean([pd.Series(preds).rank() / len(preds) for preds in lgb_predictions], axis=0)
rank_xgb_avg = np.mean([pd.Series(preds).rank() / len(preds) for preds in xgb_predictions], axis=0)

# Pondération (60% LGBM, 40% XGBoost)
ultimate_rank = (rank_lgb_avg * 0.6) + (rank_xgb_avg * 0.4)

sub_ensemble = pd.DataFrame({'id': test['id'], 'target': ultimate_rank})
out_file = "dataset/submission_v_SEED_AVG_ENHANCED.csv"
sub_ensemble.to_csv(out_file, index=False)

print(f"Fichier {out_file} généré avec succès !")
