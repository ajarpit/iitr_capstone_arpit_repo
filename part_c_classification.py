import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

def run_part_c():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # 1. Stratified Split (840 / 180 / 180)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Helper evaluation function
    def get_metrics_dict(name, y_true, y_pred, pos_prec_zero=False):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = 0.0 if pos_prec_zero else precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {
            "model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(f1), 4)
        }

    # 2. Candidate Classifiers Evaluation
    # Blind Baseline
    blind_preds = np.zeros_like(y_val)
    models_summary = [get_metrics_dict("blind_all_on_time", y_val, blind_preds, pos_prec_zero=True)]

    # Logistic Regression Pipeline
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    models_summary.append(get_metrics_dict("logistic_regression", y_val, lr_pipe.predict(X_val)))

    # Decision Tree Tuning for Depth
    best_dt_d = None
    best_dt_f1 = -1.0
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        f1_v = f1_score(y_val, dt.predict(X_val), zero_division=0)
        if f1_v > best_dt_f1:
            best_dt_f1 = f1_v
            best_dt_d = d

    dt_best = DecisionTreeClassifier(criterion="gini", max_depth=best_dt_d, random_state=42)
    dt_best.fit(X_train, y_train)
    models_summary.append(get_metrics_dict(f"decision_tree_depth_{best_dt_d}", y_val, dt_best.predict(X_val)))

    # Default Random Forest
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    models_summary.append(get_metrics_dict("random_forest_default", y_val, rf_def.predict(X_val)))

    pd.DataFrame(models_summary).to_csv("outputs/part_c_models.csv", index=False)

    # 3. Feature Importance & SelectKBest Signal Analysis
    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_

    rf_def_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def_all.fit(X_train, y_train)
    rf_imps = rf_def_all.feature_importances_

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": f_scores,
        "rf_importance": rf_imps,
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
    })
    feature_scores_df["f_rank"] = feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    feature_scores_df = feature_scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feature_scores_df.sort_values(by="f_rank", inplace=True)
    feature_scores_df["f_score"] = feature_scores_df["f_score"].round(4)
    feature_scores_df["rf_importance"] = feature_scores_df["rf_importance"].round(4)
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

    # SelectK experiment
    selectk_records = []
    for k in [5, 7, 10, 12, 17]:
        k_pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        k_pipe.fit(X_train, y_train)
        val_f1 = f1_score(y_val, k_pipe.predict(X_val), zero_division=0)
        selectk_records.append({"k": k, "val_f1": round(float(val_f1), 4)})
    pd.DataFrame(selectk_records).to_csv("outputs/part_c_selectk.csv", index=False)

    # 4. Hyperparameter Tuning (k=7)
    k7_selector = SelectKBest(f_classif, k=7)
    X_train_k7 = k7_selector.fit_transform(X_train, y_train)
    X_val_k7 = k7_selector.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }

    # Grid Search
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy")
    gs.fit(X_train_k7, y_train)
    gs_val_f1 = f1_score(y_val, gs.best_estimator_.predict(X_val_k7), zero_division=0)

    # Randomized Search
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42, scoring="accuracy")
    rs.fit(X_train_k7, y_train)
    rs_val_f1 = f1_score(y_val, rs.best_estimator_.predict(X_val_k7), zero_division=0)

    # Untuned Forest on same 7 features
    def_rf_k7 = RandomForestClassifier(n_estimators=100, random_state=42)
    def_rf_k7.fit(X_train_k7, y_train)
    def_val_f1 = f1_score(y_val, def_rf_k7.predict(X_val_k7), zero_division=0)

    tuning_df = pd.DataFrame([
        {
            "search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
            "best_params": str(gs.best_params_), "best_cv_accuracy": round(float(gs.best_score_), 4),
            "val_f1": round(float(gs_val_f1), 4)
        },
        {
            "search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
            "best_params": str(rs.best_params_), "best_cv_accuracy": round(float(rs.best_score_), 4),
            "val_f1": round(float(rs_val_f1), 4)
        },
        {
            "search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
            "best_params": "{'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1}",
            "best_cv_accuracy": "", "val_f1": round(float(def_val_f1), 4)
        }
    ])
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

    # 5. Winning Configuration Selection
    # Highest val_f1 (ties: default -> grid -> randomized)
    best_row_idx = tuning_df["val_f1"].idxmax()
    winning_search_name = tuning_df.loc[best_row_idx, "search"]

    if winning_search_name == "default_forest_same_7_features":
        final_params = {"random_state": 42}
    elif winning_search_name == "grid_search":
        final_params = {**gs.best_params_, "random_state": 42}
    else:
        final_params = {**rs.best_params_, "random_state": 42}

    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(**final_params))
    ])
    final_model.fit(X_train, y_train)

    # 6. Threshold Analysis
    val_probs = final_model.predict_proba(X_val)[:, 1]
    thresholds_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_records = []
    chosen_threshold = 0.1

    for t in thresholds_list:
        preds_t = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds_t).ravel()
        acc = accuracy_score(y_val, preds_t)
        prec = precision_score(y_val, preds_t, zero_division=0)
        rec = recall_score(y_val, preds_t, zero_division=0)
        f1 = f1_score(y_val, preds_t, zero_division=0)
        
        threshold_records.append({
            "threshold": round(t, 1), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(f1), 4)
        })

    # Ops rule: highest t with validation recall >= 0.80
    for r in threshold_records:
        if r["recall"] >= 0.80 and r["threshold"] > chosen_threshold:
            chosen_threshold = r["threshold"]

    pd.DataFrame(threshold_records).to_csv("outputs/part_c_thresholds.csv", index=False)

    # 7. Final Test Evaluation & Configuration Export
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds_final = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds_final).ravel()
    
    test_eval_df = pd.DataFrame([{
        "threshold": round(chosen_threshold, 1),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(float(accuracy_score(y_test, test_preds_final)), 4),
        "precision": round(float(precision_score(y_test, test_preds_final, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_preds_final, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, test_preds_final, zero_division=0)), 4)
    }])
    test_eval_df.to_csv("outputs/part_c_final_test.csv", index=False)

    rf_obj = final_model.named_steps["model"]
    config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": getattr(rf_obj, "n_estimators", 100),
        "max_depth": "unlimited" if getattr(rf_obj, "max_depth", None) is None else getattr(rf_obj, "max_depth"),
        "min_samples_leaf": getattr(rf_obj, "min_samples_leaf", 1),
        "threshold": round(chosen_threshold, 1),
        "chosen_from": winning_search_name
    }])
    config_df.to_csv("outputs/part_c_final_config.csv", index=False)

if __name__ == "__main__":
    run_part_c()