"""
WNS Analytics Wizard 2018 - Employee Promotion Prediction
Binary classification, evaluated on F1 score.

Usage:
    1. Place train.csv and test.csv in the same folder as this script
       (download them from the "Data" section of the competition page).
    2. Install deps:  pip install pandas numpy scikit-learn lightgbm
    3. Run:  python promotion_prediction.py
    4. Output: submission.csv (employee_id, is_promoted)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# -----------------------------
# 1. Load data
# -----------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["employee_id"].copy()

TARGET = "is_promoted"
train_len = len(train)

# Combine for consistent preprocessing
full = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)

# -----------------------------
# 2. Handle missing values
# -----------------------------
# Flag employees with no previous rating (usually length_of_service == 1)
full["no_previous_rating"] = full["previous_year_rating"].isna().astype(int)
full["previous_year_rating"] = full["previous_year_rating"].fillna(0)

full["education"] = full["education"].fillna("Unknown")

# -----------------------------
# 3. Feature engineering
# -----------------------------
full["total_score"] = (
    full["KPIs_met >80%"] + full["awards_won?"] + (full["previous_year_rating"] >= 4).astype(int)
)

full["avg_training_score_x_trainings"] = full["avg_training_score"] * full["no_of_trainings"]

full["service_per_age"] = full["length_of_service"] / full["age"]

full["rating_x_kpi"] = full["previous_year_rating"] * full["KPIs_met >80%"]

# Department-wise mean training score (computed on full set; simple, not leak-proof
# but fine as a baseline -- for a stricter approach, compute inside CV folds only)
dept_avg = full.groupby("department")["avg_training_score"].transform("mean")
full["dept_avg_training_score"] = dept_avg

# -----------------------------
# 4. Encode categoricals
# -----------------------------
cat_cols = ["department", "region", "education", "gender", "recruitment_channel"]

for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

# -----------------------------
# 5. Split back into train/test
# -----------------------------
feature_cols = [c for c in full.columns if c != "employee_id"]

X = full.iloc[:train_len][feature_cols].reset_index(drop=True)
y = train[TARGET].reset_index(drop=True)
X_test = full.iloc[train_len:][feature_cols].reset_index(drop=True)

# -----------------------------
# 6. Cross-validated training with threshold tuning
# -----------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# scale_pos_weight to handle class imbalance
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "verbosity": -1,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

    fold_f1 = f1_score(y_val, (oof_preds[val_idx] > 0.5).astype(int))
    print(f"Fold {fold + 1} F1 (thresh=0.5): {fold_f1:.4f}")

# -----------------------------
# 7. Threshold tuning on OOF predictions
# -----------------------------
best_thresh, best_f1 = 0.5, 0.0
for thresh in np.arange(0.1, 0.6, 0.01):
    f1 = f1_score(y, (oof_preds > thresh).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f"\nBest OOF F1: {best_f1:.4f} at threshold {best_thresh:.2f}")

# -----------------------------
# 8. Generate submission
# -----------------------------
final_preds = (test_preds > best_thresh).astype(int)

submission = pd.DataFrame({"employee_id": test_ids, "is_promoted": final_preds})
submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv")
print(submission["is_promoted"].value_counts())
