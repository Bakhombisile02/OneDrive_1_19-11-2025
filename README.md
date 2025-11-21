# INFO411 Assignment 2: Parkinson's Disease Detection

This project implements a machine learning pipeline for detecting Parkinson's Disease from speech features. It includes data visualization, baseline modeling, feature selection (mRMR), and advanced nested cross-validation with ensemble methods.

## 🚨 Update for Ryan and Stacy

We have performed a major code review and refactoring to improve maintainability and testing.

*   **Where is `src/modeling.py`?**
    It has been split into two files to separate concerns:
    *   `src/feature_selection.py`: Contains `mrmr_rank` and discretization logic.
    *   `src/training.py`: Contains all model pipelines, `nested_cv_run`, and task runners (`run_task_2`, `run_task_3_stacey`, etc.).
*   **New Tests**: We now have a `tests/` directory. Please run tests before pushing changes.
*   **Dependencies**: `requirements.txt` now uses pinned versions for reproducibility.

## Project Structure

```text
.
├── assignment.py           # Main entry point
├── download_data.py        # Data downloader script
├── requirements.txt        # Pinned dependencies
├── src/
│   ├── config.py           # Configuration (Env vars)
│   ├── data.py             # Data loading & preprocessing
│   ├── evaluation.py       # Metrics & CV splitting
│   ├── feature_selection.py# mRMR & discretization (Refactored)
│   ├── training.py         # Model training & Task runners (Refactored)
│   └── visualization.py    # PCA, t-SNE, Fisher Ratio
├── tests/                  # Unit tests (New!)
└── data/                   # Dataset storage
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data**:
    ```bash
    python download_data.py
    ```

## Configuration

You can configure the project using environment variables (optional).

*   `DATA_PATH`: Path to the CSV file (default: `data/Parkinsons_Speech-Features.csv`)
*   `SEED`: Random seed for reproducibility (default: `42`)

Example:
```bash
export SEED=123
python assignment.py
```

## Running the Analysis

Run the full pipeline (Tasks 1-4):

```bash
python assignment.py --flow both
```

Run specific flows:

```bash
# Run only Stacey's pipeline (Task 3a/4a)
python assignment.py --flow stacey

# Or use the convenience script:
./run_stacey.sh

# Run only Siya's pipeline (Task 3b/4b)
python assignment.py --flow siya
```

### Outputs

Results are saved to the `results/` directory:
*   `stacey_task3_results.csv`: Metrics for Logistic Regression vs Random Forest.
*   `stacey_task4_results.csv`: Metrics for the Soft Voting Ensemble.

### CLI Options

*   `--siya-device [auto|cpu]`: Force CPU usage or allow auto-detection of GPU (CUDA/Metal).
*   `--siya-enhanced`: Enable broader hyperparameter grids and calibration.
*   `--siya-parallel-configs`: Parallelize inner CV loops (memory intensive).

## Testing

We use `pytest` for unit testing.

**Run all tests:**
```bash
python -m pytest tests/
```

**Run specific test file:**
```bash
python -m pytest tests/test_feature_selection.py
```

Please ensure all tests pass before submitting changes.
