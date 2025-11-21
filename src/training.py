import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED
from src.evaluation import compute_metrics, get_stratified_group_folds
from src.feature_selection import mrmr_rank
from src.visualization import plot_pr_curves

# --- XGBoost Setup ---
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"XGBoost import failed: {e}")
    print("Falling back to sklearn GradientBoostingClassifier.")
    XGBOOST_AVAILABLE = False


def detect_cuda_device():
    """Detect CUDA availability"""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=2,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def get_xgboost_device_params(device_preference="auto"):
    """Return hardware-aware parameters for XGBoost."""
    if not XGBOOST_AVAILABLE:
        return {}

    prefers_gpu = device_preference != "cpu"
    if prefers_gpu and detect_cuda_device():
        print("Using CUDA for XGBoost")
        return {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
        }

    print("Using CPU execution for XGBoost")
    return {'tree_method': 'hist', 'predictor': 'auto', 'n_jobs': -1}


# --- Task 2: Classification Evaluation ---

def evaluate_all_schemes(X_schemes, y, subject_id, n_folds=5, seed=SEED):
    print("\n--- Task 2: Classification Evaluation ---")

    # Define models with Pipelines to ensure scaling happens INSIDE the fold
    models = {
        "Decision Tree": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=10, random_state=seed))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=seed))
        ]),
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', n_jobs=1))
        ])
    }

    results = []
    # Use StratifiedGroupKFold for proper evaluation
    fold_indices = get_stratified_group_folds(
        np.zeros(len(y)), y, subject_id, n_folds, seed
    )

    for scheme_name, X in X_schemes.items():
        n_features = X.shape[1]
        print(f"Evaluating: {scheme_name} ({n_features} features)")

        for model_name, model in models.items():
            fold_metrics = []

            for fold in range(n_folds):
                test_mask = fold_indices == fold
                train_mask = ~test_mask

                if np.sum(test_mask) == 0:
                    continue

                X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
                X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] \
                    if hasattr(model, "predict_proba") else None

                metrics = compute_metrics(y_test, y_pred, y_prob)
                fold_metrics.append(metrics)

            # Aggregate results
            if not fold_metrics:
                continue

            avg_metrics = {
                k: np.mean([m[k] for m in fold_metrics])
                for k in fold_metrics[0]
            }
            std_metrics = {
                k: np.std([m[k] for m in fold_metrics])
                for k in fold_metrics[0]
            }

            res_entry = {
                "Scheme": scheme_name,
                "Model": model_name,
                "N_Features": n_features,
                **{f"{k}_mean": v for k, v in avg_metrics.items()},
                **{f"{k}_std": v for k, v in std_metrics.items()}
            }
            results.append(res_entry)
            print(
                f"  {model_name}: Acc={avg_metrics['Accuracy']:.4f} "
                f"+/- {std_metrics['Accuracy']:.4f}"
            )

    results_df = pd.DataFrame(results)
    return results_df


def run_task_2(schemes, y, df):
    subject_id = df.iloc[:, 0]  # Subject ID is column 1 (index 0)
    results_df = evaluate_all_schemes(schemes, y, subject_id)

    # Find best model per scheme
    print("\nBest Results by Scheme:")
    if not results_df.empty:
        best_by_scheme = results_df.loc[
            results_df.groupby("Scheme")["Accuracy_mean"].idxmax()
        ]
        print(
            best_by_scheme[["Scheme", "Model", "Accuracy_mean", "F1_mean"]]
        )

    return results_df


def best_threshold_mcc(y_true, y_prob):
    thresholds = np.arange(0, 1.01, 0.01)
    mccs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        mccs.append(matthews_corrcoef(y_true, y_pred))

    best_idx = np.argmax(mccs)
    return thresholds[best_idx], mccs[best_idx], thresholds, mccs


# --- Task 3: Nested CV ---

def _evaluate_config(config, X_train_outer, y_train_outer, inner_folds, n_inner):
    model_type, k, params = config
    inner_scores = []

    for inner_f in range(n_inner):
        val_mask = inner_folds == inner_f
        tr_mask = ~val_mask

        if np.sum(val_mask) == 0:
            continue

        X_tr_in, y_tr_in = X_train_outer[tr_mask], y_train_outer[tr_mask]
        X_val_in, y_val_in = X_train_outer[val_mask], y_train_outer[val_mask]

        # Standardize inside inner fold
        scaler = StandardScaler()
        X_tr_in_scaled = scaler.fit_transform(X_tr_in)
        X_val_in_scaled = scaler.transform(X_val_in)

        selected_feats = mrmr_rank(X_tr_in_scaled, y_tr_in, k=k)
        X_tr_in_sel = X_tr_in_scaled[:, selected_feats]
        X_val_in_sel = X_val_in_scaled[:, selected_feats]

        if model_type == 'svm':
            model = SVC(
                probability=True, class_weight='balanced', **params
            )
        else:
            if XGBOOST_AVAILABLE:
                model = XGBClassifier(eval_metric='logloss', **params)
            else:
                allowed = [
                    'learning_rate', 'max_depth', 'n_estimators', 'subsample'
                ]
                clean_params = {
                    k_: v_ for k_, v_ in params.items() if k_ in allowed
                }
                model = GradientBoostingClassifier(**clean_params)

        model.fit(X_tr_in_sel, y_tr_in)
        y_prob = model.predict_proba(X_val_in_sel)[:, 1]

        precision, recall, _ = precision_recall_curve(y_val_in, y_prob)
        auprc = auc(recall, precision)
        inner_scores.append(auprc)

    mean_score = np.mean(inner_scores) if inner_scores else 0
    return (model_type, k, params, mean_score)


def nested_cv_run(
    X,
    y,
    subject_id,
    n_outer=3,
    n_inner=3,
    seed=SEED,
    k_list=None,
    svm_grid=None,
    gbdt_grid=None,
    device_preference="auto",
    parallel_configs=False,
    enhanced=False,
    ensemble_size=1,
    calibration_method=None,
):
    print("\n--- Task 3: Nested Cross-Validation (Optimized) ---")

    # Use StratifiedGroupKFold for outer loop
    outer_folds = get_stratified_group_folds(X, y, subject_id, n_outer, seed)

    X = np.array(X)
    y = np.array(y)
    subject_id = np.array(subject_id)

    results = []

    if k_list is None:
        k_list = [20, 50]
    k_list = [k for k in k_list if k > 0]
    if not k_list:
        raise ValueError("k_list must contain at least one positive integer")
    if svm_grid is None:
        svm_grid = [
            {'C': 1.0, 'gamma': 0.05},
            {'C': 1.0, 'gamma': 0.1},
            {'C': 10.0, 'gamma': 0.05},
            {'C': 10.0, 'gamma': 0.1},
        ]

    xgb_device_params = get_xgboost_device_params(device_preference)
    if gbdt_grid is None:
        gbdt_grid = [
            {
                'learning_rate': 0.05,
                'max_depth': 5,
                'n_estimators': 200,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                **xgb_device_params,
            },
            {
                'learning_rate': 0.1,
                'max_depth': 4,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                **xgb_device_params,
            },
        ]

    if enhanced:
        ensemble_size = max(ensemble_size, 3)
        if calibration_method is None:
            calibration_method = 'isotonic'
        k_list = sorted(set(k_list + [30, 75, 100]))
        svm_grid = svm_grid + [
            {'C': 5.0, 'gamma': 0.05},
            {'C': 5.0, 'gamma': 0.1},
            {'C': 20.0, 'gamma': 0.05},
        ]
        gbdt_grid = gbdt_grid + [
            {
                'learning_rate': 0.03,
                'max_depth': 6,
                'n_estimators': 400,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                **xgb_device_params,
            },
            {
                'learning_rate': 0.08,
                'max_depth': 4,
                'n_estimators': 500,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                **xgb_device_params,
            },
        ]

    for f in range(n_outer):
        print(f"\nOUTER FOLD {f+1}/{n_outer}")
        test_mask = outer_folds == f
        train_mask = ~test_mask

        if np.sum(test_mask) == 0:
            print("  Skipping fold (empty test set)")
            continue

        X_train_outer, y_train_outer = X[train_mask], y[train_mask]
        X_test_outer, y_test_outer = X[test_mask], y[test_mask]
        subj_train_outer = subject_id[train_mask]

        # Inner CV for Model Selection (StratifiedGroupKFold)
        inner_folds = get_stratified_group_folds(
            X_train_outer, y_train_outer, subj_train_outer, n_inner, seed
        )

        configs = []
        for k in k_list:
            for p in svm_grid:
                configs.append(('svm', k, p))
            for p in gbdt_grid:
                configs.append(('gbdt', k, p))

        print(f"  Running Inner CV on {len(configs)} configurations...")

        if parallel_configs and len(configs) > 1:
            evaluated = Parallel(n_jobs=-1)(
                delayed(_evaluate_config)(
                    cfg, X_train_outer, y_train_outer, inner_folds, n_inner
                ) for cfg in configs
            )
        else:
            evaluated = [
                _evaluate_config(
                    cfg, X_train_outer, y_train_outer, inner_folds, n_inner
                ) for cfg in configs
            ]

        evaluated_sorted = sorted(evaluated, key=lambda x: x[3], reverse=True)
        if not evaluated_sorted:
            raise RuntimeError(
                "No configurations were successfully evaluated in inner CV."
            )
        best_model_type, best_k, best_params, best_score = evaluated_sorted[0]
        ensemble_configs = evaluated_sorted[:ensemble_size]
        print(
            f"  Best Config: ({best_model_type}, {best_k}, {best_params}) "
            f"(AUPRC={best_score:.4f})"
        )

        # Retrain on full outer train set
        ensemble_probs = []
        config_summaries = []

        # Standardize outer train/test
        scaler_outer = StandardScaler()
        X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler_outer.transform(X_test_outer)

        for model_type, k, params, _ in ensemble_configs:
            selected_feats = mrmr_rank(
                X_train_outer_scaled, y_train_outer, k=k
            )
            X_train_sel = X_train_outer_scaled[:, selected_feats]
            X_test_sel = X_test_outer_scaled[:, selected_feats]

            if model_type == 'svm':
                base_model = SVC(
                    probability=True, class_weight='balanced', **params
                )
            else:
                if XGBOOST_AVAILABLE:
                    base_model = XGBClassifier(eval_metric='logloss', **params)
                else:
                    allowed = [
                        'learning_rate', 'max_depth', 'n_estimators',
                        'subsample', 'colsample_bytree'
                    ]
                    clean_params = {
                        k_: v_ for k_, v_ in params.items() if k_ in allowed
                    }
                    base_model = GradientBoostingClassifier(**clean_params)

            if calibration_method:
                # Create group-aware CV iterator for calibration
                calib_cv = list(StratifiedGroupKFold(
                    n_splits=3, shuffle=True, random_state=seed
                ).split(X_train_sel, y_train_outer, subj_train_outer))
                model = CalibratedClassifierCV(
                    base_model, method=calibration_method, cv=calib_cv
                )
                model.fit(X_train_sel, y_train_outer)
            else:
                base_model.fit(X_train_sel, y_train_outer)
                model = base_model

            y_prob_single = model.predict_proba(X_test_sel)[:, 1]
            ensemble_probs.append(y_prob_single)
            config_summaries.append(
                {'model': model_type, 'k': k, 'params': params}
            )

        if not ensemble_probs:
            raise RuntimeError(
                "Ensemble produced no predictions; check configuration."
            )
        y_prob = np.mean(ensemble_probs, axis=0) \
            if len(ensemble_probs) > 1 else ensemble_probs[0]

        # Evaluate
        precision, recall, _ = precision_recall_curve(y_test_outer, y_prob)
        auprc = auc(recall, precision)

        tau_opt, mcc_val, thresholds, mccs = best_threshold_mcc(
            y_test_outer, y_prob
        )

        print(
            f"  [Outer Fold {f+1}] AUPRC={auprc:.4f}, MCC={mcc_val:.4f}, "
            f"Opt Tau={tau_opt:.2f}"
        )

        results.append({
            "Fold": f+1,
            "Model": best_model_type,
            "k": best_k,
            "AUPRC": auprc,
            "MCC": mcc_val,
            "Threshold": tau_opt,
            "PR_Curve": (recall, precision),
            "MCC_Curve": (thresholds, mccs),
            "Ensemble": config_summaries,
        })

    return results


# --- Task 3 & 4 Runners ---

def run_task_3_stacey(X_all, y, subject_id, seed=SEED, output_dir="results"):
    print("\n--- Task 3 (Stacey): MI + Elastic Net ---")

    print("  Applying Mutual Information Filter (Top 150)...")
    n_folds = 5
    fold_indices = get_stratified_group_folds(
        X_all, y, subject_id, n_folds, seed
    )

    # Pipeline: Scaler -> MI (150) -> ElasticNet LR
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mi', SelectKBest(mutual_info_classif, k=150)),
        ('clf', LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=5000, class_weight='balanced', random_state=seed
        ))
    ])

    # Random Forest (Comparison)
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=seed
        ))
    ])

    models = {
        "LogisticRegression_ElasticNet": lr_pipeline,
        "RandomForest": rf_pipeline
    }

    results = []

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        fold_metrics = []
        pr_results = []

        for fold in range(n_folds):
            test_mask = fold_indices == fold
            train_mask = ~test_mask

            if np.sum(test_mask) == 0:
                continue

            X_train, y_train = X_all[train_mask], y[train_mask]
            X_test, y_test = X_all[test_mask], y[test_mask]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)
            fold_metrics.append(metrics)
            
            # Compute PR Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            auprc = auc(recall, precision)
            pr_results.append({
                'Fold': fold + 1,
                'PR_Curve': (recall, precision),
                'AUPRC': auprc
            })

        avg_metrics = {
            k: np.mean([m[k] for m in fold_metrics])
            for k in fold_metrics[0]
        }
        print(
            f"    Acc={avg_metrics['Accuracy']:.4f}, "
            f"F1={avg_metrics['F1']:.4f}, "
            f"Recall={avg_metrics['Sensitivity']:.4f}"
        )
        results.append({'Model': name, **avg_metrics})
        
        # Plot PR Curves
        plot_pr_curves(pr_results, title_prefix=f"Task 3 - {name}", save_dir=output_dir)

    return pd.DataFrame(results), models


def run_task_4_stacey(X_all, y, subject_id, trained_models_task3a):
    print("\n--- Task 4 (Stacey): Soft Voting Ensemble ---")

    # Recreate pipelines for VotingClassifier
    rf = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=SEED
    )
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('mi', SelectKBest(mutual_info_classif, k=150)),
        ('clf', LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=5000, class_weight='balanced', random_state=SEED
        ))
    ])
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=14))
    ])

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('knn', knn_pipe)],
        voting='soft'
    )

    # Evaluate Ensemble
    n_folds = 5
    fold_indices = get_stratified_group_folds(
        X_all, y, subject_id, n_folds, SEED
    )
    fold_metrics = []

    for fold in range(n_folds):
        test_mask = fold_indices == fold
        train_mask = ~test_mask

        if np.sum(test_mask) == 0:
            continue

        X_train, y_train = X_all[train_mask], y[train_mask]
        X_test, y_test = X_all[test_mask], y[test_mask]

        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_prob = ensemble.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics.append(metrics)

    avg_metrics = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in fold_metrics[0]
    }
    print(
        f"  Ensemble: Acc={avg_metrics['Accuracy']:.4f}, "
        f"F1={avg_metrics['F1']:.4f}, "
        f"Recall={avg_metrics['Sensitivity']:.4f}"
    )
    return avg_metrics


def run_task_3_siya(schemes, y, df, output_dir="results", **siya_cv_kwargs):
    print("\n--- Task 3 (Siya): mRMR + Nested CV ---")
    # Combine all features
    X_all_np = np.hstack([X.values for X in schemes.values()])
    subject_id = df.iloc[:, 0]

    # Run Nested CV
    results = nested_cv_run(X_all_np, y, subject_id, **siya_cv_kwargs)
    
    # Plot PR Curves
    plot_pr_curves(results, title_prefix="Task 3 (Siya) - Nested CV", save_dir=output_dir)
    
    return results


def run_task_4_siya(X_all, y, df, seed=SEED, output_dir="results", **siya_cv_kwargs):
    print("\n--- Task 4 (Siya): Gender-Specific Models ---")
    subject_id = df.iloc[:, 0]
    gender = df.iloc[:, 1]

    g_counts = gender.value_counts()
    print(f"  Gender counts: {g_counts.to_dict()}")

    results = {}
    for g_val in g_counts.index:
        print(f"  Running Nested CV for Gender {g_val}...")
        mask = gender == g_val
        X_g = X_all[mask]
        y_g = y[mask]
        subj_g = subject_id[mask]

        if len(np.unique(y_g)) < 2:
            print("    Skipping: Only one class present.")
            continue

        # Run Nested CV
        results_g = nested_cv_run(
            X_g,
            y_g,
            subj_g,
            seed=seed,
            **siya_cv_kwargs,
        )

        avg_auprc = np.mean([r['AUPRC'] for r in results_g])
        avg_mcc = np.mean([r['MCC'] for r in results_g])
        print(
            f"    Gender {g_val} Results: Mean AUPRC={avg_auprc:.4f}, "
            f"Mean MCC={avg_mcc:.4f}"
        )
        results[g_val] = results_g
        
        # Plot PR Curves
        plot_pr_curves(results_g, title_prefix=f"Task 4 (Siya) - Gender {g_val}", save_dir=output_dir)

    return results
