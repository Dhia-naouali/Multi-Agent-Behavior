import gc
import json
import torch
import optuna

import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold



def tune_threshold(oof_action, y_action):
    thresholds = np.arange(0, 1.005, 0.005)
    scores = [f1_score(y_action, (oof_action >= th), zero_division=0) for th in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx]


def train_validate(
    lab_id,
    behavior,
    indices,
    features,
    labels,
    working_dir
):
    
    result_dir = working_dir / "results" / lab_id / behavior
    result_dir.mkdir(exist_ok=True, parents=True)
    
    if labels.sum() == 0:
        with open(result_dir / "f1.txt", "w") as f:
            f.write("0.0\n")
        oof_prediction_dataframe = indices.with_columns(
            pl.Series("fold", [-1] * len(labels), dtype=pl.Int8),
            pl.Series("prediction", [0.0] * len(labels), dtype=pl.Float32),
            pl.Series("predicted_label", [0] * len(labels), dtype=pl.Int8),
        )
        oof_prediction_dataframe.write_parquet(result_dir / "oof_predictions.parquet")
        return 0.0
    
    folds = np.ones(len(labels), dtype=np.int8) * -1
    oof_predictions = np.zeros(len(labels), dtype=np.float32)
    oof_prediction_labels = np.zeros(len(labels), dtype=np.int8)
    
    scale_pos_weight = (len(labels) - labels.sum()) / labels.sum()
    
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "tree_method": "hist",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight": scale_pos_weight,
            "max_bin": trial.suggest_int("max_bin", 32, 256),
            "seed": 12,
        }
        
        cv_scores = []
        
        for train_idx, valid_idx in StratifiedGroupKFold(
                n_splits=3, 
                shuffle=True, 
                random_state=12
            ).split(
                X=features,
                y=labels,
                groups=indices.get_column("video_id"),
            ):
            X_train = features[train_idx]
            y_train = labels[train_idx]
            X_valid = features[valid_idx]
            y_valid = labels[valid_idx]
            
            dtrain = xgb.QuantileDMatrix(
                X_train.to_numpy(), 
                label=y_train.to_numpy(), 
                feature_names=features.columns, 
                max_bin=params["max_bin"]
            )
            dvalid = xgb.DMatrix(
                X_valid.to_numpy(), 
                label=y_valid.to_numpy(), 
                feature_names=features.columns
            )
            
            model = xgb.train(
                params,
                dtrain=dtrain,
                num_boost_round=250,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                verbose_eval=0,
            )
            
            fold_predictions = model.predict(dvalid)
            fold_f1 = f1_score(
                y_valid, 
                (fold_predictions >= 0.5).astype(np.int8), 
                zero_division=0
            )
            cv_scores.append(fold_f1)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(objective, n_trials=16, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "seed": 12,
    })

    
    with open(result_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    
    # full data train
    for fold, (train_idx, valid_idx) in enumerate(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
            X=features,
            y=labels,
            groups=indices.get_column("video_id"),
        )
    ):
        result_dir_fold = result_dir / f"fold_{fold}"
        result_dir_fold.mkdir(exist_ok=True, parents=True)
        
        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_valid = features[valid_idx]
        y_valid = labels[valid_idx]
        
        dtrain = xgb.QuantileDMatrix(
            X_train.to_numpy(), 
            label=y_train.to_numpy(), 
            feature_names=features.columns, 
            max_bin=best_params["max_bin"]
        )
        dvalid = xgb.DMatrix(
            X_valid.to_numpy(), 
            label=y_valid.to_numpy(), 
            feature_names=features.columns
        )
        
        evals_result = {}
        early_stopping_callback = xgb.callback.EarlyStopping(
            rounds=10,
            metric_name="logloss",
            data_name="valid",
            maximize=False,
            save_best=True,
        )
        
        model = xgb.train(
            best_params,
            dtrain=dtrain,
            num_boost_round=250,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            callbacks=[early_stopping_callback],
            evals_result=evals_result,
            verbose_eval=0,
        )
        
        fold_predictions = model.predict(dvalid)
        threshold = tune_threshold(fold_predictions, y_valid)
        
        folds[valid_idx] = fold
        oof_predictions[valid_idx] = fold_predictions
        oof_prediction_labels[valid_idx] = (fold_predictions >= threshold).astype(np.int8)
        
        model.save_model(result_dir_fold / "model.json")
        
        with open(result_dir_fold / "threshold.txt", "w") as f:
            f.write(f"{threshold}\n")
        

        xgb.plot_importance(model, max_num_features=20, importance_type="gain", values_format="{v:.2f}")
        plt.tight_layout()
        plt.savefig(result_dir_fold / "feature_importance.png")
        plt.close()
        

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(evals_result["train"]["logloss"], label="Train")
        ax.plot(evals_result["valid"]["logloss"], label="Valid")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(result_dir_fold / "metric.png")
        plt.close()
        
        gc.collect()
    
    # oof preds
    oof_prediction_dataframe = indices.with_columns(
        pl.Series("fold", folds, dtype=pl.Int8),
        pl.Series("prediction", oof_predictions, dtype=pl.Float32),
        pl.Series("predicted_label", oof_prediction_labels, dtype=pl.Int8),
    )
    
    f1 = f1_score(labels, oof_prediction_labels, zero_division=0)
    
    with open(result_dir / "f1.txt", "w") as f:
        f.write(f"{f1}\n")
    
    oof_prediction_dataframe.write_parquet(result_dir / "oof_predictions.parquet")

    return f1