import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

def run_part_b():
    os.makedirs("outputs", exist_ok=True)
    
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # 1. Train / Val / Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 2. Leakage-free scaling verification
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
    test_max_abs_mean = float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))

    checks = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(train_max_abs_mean, 9)),
        ("test_scaled_max_abs_mean", round(test_max_abs_mean, 4)),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)

    # 3. Tree Depth Diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_records = []
    best_depth = None
    best_val_rmse = float("inf")

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        
        tr_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
        v_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
        
        depth_records.append({
            "max_depth": "unlimited" if d is None else str(d),
            "train_rmse": round(float(tr_rmse), 4),
            "val_rmse": round(float(v_rmse), 4)
        })
        
        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_depth = d

    pd.DataFrame(depth_records).to_csv("outputs/part_b_tree_depth.csv", index=False)

    # 4. Candidate Models
    models = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    metrics = []

    # Baseline
    mean_val_pred = np.full_like(y_val, fill_value=y_train.mean())
    b_mae = mean_absolute_error(y_val, mean_val_pred)
    b_mse = mean_squared_error(y_val, mean_val_pred)
    metrics.append({
        "model": "mean_baseline", "split": "val",
        "MAE": round(float(b_mae), 4), "MSE": round(float(b_mse), 4),
        "RMSE": round(float(np.sqrt(b_mse)), 4), "R2": round(float(r2_score(y_val, mean_val_pred)), 4)
    })

    val_rmses = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        metrics.append({
            "model": name, "split": "val",
            "MAE": round(float(mae), 4), "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4), "R2": round(float(r2), 4)
        })
        val_rmses[name] = rmse

    # 5. Learning Rate Experiment
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        preds = gb.predict(X_val)
        v_rmse = np.sqrt(mean_squared_error(y_val, preds))
        v_r2 = r2_score(y_val, preds)
        lr_records.append({
            "learning_rate": lr,
            "val_rmse": round(float(v_rmse), 4),
            "val_r2": round(float(v_r2), 4)
        })
    pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    # 6. CV on top 2 models
    sorted_candidates = sorted(val_rmses.keys(), key=lambda k: val_rmses[k])
    top_2 = sorted_candidates[:2]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_records = []
    for name in top_2:
        scores = cross_val_score(models[name], X_train_val, y_train_val, cv=kf)
        cv_records.append({
            "model": name,
            "mean_r2": round(float(np.mean(scores)), 4),
            "std_r2": round(float(np.std(scores)), 4)
        })
    pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)

    # 7. Final Model Evaluation on Test Set
    winning_model_name = sorted_candidates[0]
    winning_model = models[winning_model_name]
    
    test_preds = winning_model.predict(X_test)
    t_mae = mean_absolute_error(y_test, test_preds)
    t_mse = mean_squared_error(y_test, test_preds)
    t_rmse = np.sqrt(t_mse)
    t_r2 = r2_score(y_test, test_preds)

    metrics.append({
        "model": winning_model_name, "split": "test",
        "MAE": round(float(t_mae), 4), "MSE": round(float(t_mse), 4),
        "RMSE": round(float(t_rmse), 4), "R2": round(float(t_r2), 4)
    })

    pd.DataFrame(metrics).to_csv("outputs/part_b_metrics.csv", index=False)

if __name__ == "__main__":
    run_part_b()