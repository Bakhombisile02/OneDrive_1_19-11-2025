import argparse
import numpy as np
import warnings
from src.config import DATA_PATH
from src.data import load_data
from src.visualization import run_task_1
from src.training import (
    run_task_2,
    run_task_3_stacey,
    run_task_4_stacey,
    run_task_3_siya,
    run_task_4_siya
)

# warnings.filterwarnings('ignore')

def run_stacey_flow(X_all_np, y, subject_id):
    """Execute Stacey's Task 3 and Task 4 pipeline."""
    print("\n=== Running Stacey flow (Tasks 3 & 4) ===")
    task3_results, trained_models_3a = run_task_3_stacey(X_all_np, y, subject_id)
    task4_results = run_task_4_stacey(X_all_np, y, subject_id, trained_models_3a)
    return task3_results, task4_results

def run_siya_flow(schemes, y, df, X_all_np, siya_options=None):
    """Execute Siya's Task 3 and Task 4 pipeline."""
    print("\n=== Running Siya flow (Tasks 3 & 4) ===")
    siya_options = siya_options or {}
    task3_results = run_task_3_siya(schemes, y, df, **siya_options)
    task4_results = run_task_4_siya(X_all_np, y, df, **siya_options)
    return task3_results, task4_results

def parse_args():
    parser = argparse.ArgumentParser(description="INFO411 Assignment 2 runner")
    parser.add_argument(
        "--flow",
        choices=["stacey", "siya", "both"],
        default="both",
        help="Choose which advanced pipelines (Task 3/4) to execute after Tasks 1-2.",
    )
    parser.add_argument(
        "--siya-device",
        choices=["auto", "cpu"],
        default="auto",
        help="Hardware preference for Siya's gradient boosting models (auto tries CUDA/Metal).",
    )
    parser.add_argument(
        "--siya-k",
        type=int,
        nargs="+",
        default=[20, 50],
        help="Feature counts (k) to evaluate for Siya's mRMR selector.",
    )
    parser.add_argument(
        "--siya-outer-folds",
        type=int,
        default=3,
        help="Number of outer folds for Siya's nested CV.",
    )
    parser.add_argument(
        "--siya-inner-folds",
        type=int,
        default=3,
        help="Number of inner folds for Siya's nested CV.",
    )
    parser.add_argument(
        "--siya-parallel-configs",
        action="store_true",
        help="Parallelize Siya's inner-loop hyperparameter configurations (more memory intensive).",
    )
    parser.add_argument(
        "--siya-enhanced",
        action="store_true",
        help="Enable Siya enhanced mode (broader grid, ensembles, calibration).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load Data
    df = load_data(DATA_PATH)
    
    # Task 1: Visualization & Separability (Ryan)
    # Note: schemes returned here are RAW (unscaled) to prevent leakage
    schemes, y = run_task_1(df)
    
    # Task 2: Classification (Ryan)
    task2_results = run_task_2(schemes, y, df)
    
    # Prepare shared resources for Task 3 & 4 flows
    X_all_np = np.hstack([X.values for X in schemes.values()])
    subject_id = df.iloc[:, 0]

    if args.flow in ("stacey", "both"):
        task3_stacey_results, task4_stacey_results = run_stacey_flow(X_all_np, y, subject_id)

    if args.flow in ("siya", "both"):
        parallel_flag = args.siya_parallel_configs or (args.siya_device != "cpu")
        siya_options = {
            'n_outer': args.siya_outer_folds,
            'n_inner': args.siya_inner_folds,
            'k_list': args.siya_k,
            'device_preference': args.siya_device,
            'parallel_configs': parallel_flag,
            'enhanced': args.siya_enhanced,
        }
        task3_siya_results, task4_siya_results = run_siya_flow(
            schemes,
            y,
            df,
            X_all_np,
            siya_options=siya_options,
        )
