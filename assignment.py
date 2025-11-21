import argparse
import numpy as np
import pandas as pd
import os
import warnings
from src.config import DATA_PATH, SEED
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
    run_task_4_siya,
    run_shared_comparison,
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
    parser.add_argument(
        "--comparison-mode",
        action="store_true",
        help="Run shared comparison harness so Stacey and Siya flows share folds and metrics.",
    )
    parser.add_argument(
        "--comparison-schemes",
        nargs="+",
        help="Optional subset of feature schemes to include in comparison mode.",
    )
    parser.add_argument(
        "--comparison-folds",
        type=int,
        default=5,
        help="Number of grouped folds for the shared comparison harness.",
    )
    parser.add_argument(
        "--comparison-bootstrap",
        type=int,
        default=500,
        help="Bootstrap samples for comparison-mode significance tests.",
    )
    parser.add_argument(
        "--profile-comparison",
        action="store_true",
        help="Increase bootstrap iterations and enable heavier logging in comparison mode.",
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
        "task4_siya": "results/task4_siya",
        "comparison": "results/comparison",
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
    if "AllFeatures" in schemes:
        X_all_np = schemes["AllFeatures"].values
    else:
        X_all_np = np.hstack([X.values for X in schemes.values()])
    subject_id = df.iloc[:, 0]
    parallel_flag = args.siya_parallel_configs or (args.siya_device != "cpu")
    siya_options = {
        'n_outer': args.siya_outer_folds,
        'n_inner': args.siya_inner_folds,
        'k_list': args.siya_k,
        'device_preference': args.siya_device,
        'parallel_configs': parallel_flag,
        'enhanced': args.siya_enhanced,
    }

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

        if args.comparison_mode:
            comparison_bootstrap = args.comparison_bootstrap
            if args.profile_comparison:
                comparison_bootstrap = max(comparison_bootstrap, 1000)
            comparison_config = {
                'output_dir': results_dirs['comparison'],
                'schemes': args.comparison_schemes,
                'n_folds': args.comparison_folds,
                'n_boot': comparison_bootstrap,
                'seed': SEED,
            }
            comparison_artifacts = run_shared_comparison(
                schemes,
                y,
                df,
                siya_options=siya_options,
                comparison_config=comparison_config,
            )
            print(f"\n[Comparison] Artifacts: {comparison_artifacts}")
