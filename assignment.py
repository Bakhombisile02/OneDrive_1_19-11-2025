import argparse
import numpy as np
import pandas as pd
import os
import warnings
from src.config import DATA_PATH
from src.data import load_data
from src.visualization import (
    run_task_1,
    plot_task2_performance,
    plot_pr_curves
)
from src.training import (
    run_task_2,
    run_task_3_stacey,
    run_task_4_stacey,
    run_task_3_siya,
    run_task_4_siya
)

# warnings.filterwarnings('ignore')

def run_stacey_flow(X_all_np, y, subject_id, output_dir_t3, output_dir_t4):
    """Execute Stacey's Task 3 and Task 4 pipeline."""
    print("\n=== Running Stacey flow (Tasks 3 & 4) ===")
    task3_results, trained_models_3a = run_task_3_stacey(X_all_np, y, subject_id, output_dir=output_dir_t3)
    task4_results = run_task_4_stacey(X_all_np, y, subject_id, trained_models_3a)
    return task3_results, task4_results

def run_siya_flow(schemes, y, df, X_all_np, output_dir_t3, output_dir_t4, siya_options=None):
    """Execute Siya's Task 3 and Task 4 pipeline."""
    print("\n=== Running Siya flow (Tasks 3 & 4) ===")
    siya_options = siya_options or {}
    task3_results = run_task_3_siya(schemes, y, df, output_dir=output_dir_t3, **siya_options)
    task4_results = run_task_4_siya(X_all_np, y, df, output_dir=output_dir_t4, **siya_options)
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
    
    # Define and create results directories
    results_dirs = {
        "task1": "results/task1",
        "task2": "results/task2",
        "task3_stacey": "results/task3_stacey",
        "task3_siya": "results/task3_siya",
        "task4_stacey": "results/task4_stacey",
        "task4_siya": "results/task4_siya"
    }
    
    for d in results_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Load Data
    df = load_data(DATA_PATH)
    
    # Task 1: Visualization & Separability (Ryan)
    # Note: schemes returned here are RAW (unscaled) to prevent leakage
    schemes, y = run_task_1(df, output_dir=results_dirs["task1"])
    
    # Task 2: Classification (Ryan)
    task2_results = run_task_2(schemes, y, df)
    plot_task2_performance(task2_results, save_path=os.path.join(results_dirs["task2"], "task2_performance.png"))
    task2_results.to_csv(os.path.join(results_dirs["task2"], "task2_results.csv"), index=False)
    
    # Prepare shared resources for Task 3 & 4 flows
    X_all_np = np.hstack([X.values for X in schemes.values()])
    subject_id = df.iloc[:, 0]

    if args.flow in ("stacey", "both"):
        task3_stacey_results, task4_stacey_results = run_stacey_flow(
            X_all_np, y, subject_id, 
            output_dir_t3=results_dirs["task3_stacey"],
            output_dir_t4=results_dirs["task4_stacey"]
        )
        
        # Save Stacey's results
        task3_stacey_results.to_csv(os.path.join(results_dirs["task3_stacey"], "stacey_task3_results.csv"), index=False)
        pd.DataFrame([task4_stacey_results]).to_csv(os.path.join(results_dirs["task4_stacey"], "stacey_task4_results.csv"), index=False)
        print(f"\n[Saved] Stacey's results saved to {results_dirs['task3_stacey']} and {results_dirs['task4_stacey']}")

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
            output_dir_t3=results_dirs["task3_siya"],
            output_dir_t4=results_dirs["task4_siya"],
            siya_options=siya_options,
        )
        
        # Save Siya's results CSVs (plots are handled inside the functions now)
        pd.DataFrame(task3_siya_results).to_csv(os.path.join(results_dirs["task3_siya"], "siya_task3_results.csv"), index=False)
        
        # Task 4 Siya returns a dict of results per gender
        for gender, res_list in task4_siya_results.items():
            pd.DataFrame(res_list).to_csv(os.path.join(results_dirs["task4_siya"], f"siya_task4_gender_{gender}_results.csv"), index=False)
            
        print(f"\n[Saved] Siya's results saved to {results_dirs['task3_siya']} and {results_dirs['task4_siya']}")
