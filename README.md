# INFO411 Assignment 2: Parkinson's Disease Detection (Python Port)

**Welcome Ryan and Stacey!** 👋

This repository is the Python implementation of our Assignment 2 workflow. It replaces the legacy Julia notebooks (`A2_*.jl`) with a modular, production-ready Python pipeline.

If you are used to the Julia notebooks, this guide will help you navigate the new structure.

## 🗺️ Migration Guide: From Julia to Python

We have ported all logic from the Julia notebooks into a structured Python package (`src/`).

| Legacy Julia Notebook | New Python Location | Description |
| :--- | :--- | :--- |
| `A2_Task_1_and_2_Ryan_Teo.jl` | `src/visualization.py` | **Task 1**: PCA, t-SNE, Isomap, and Fisher Ratio analysis. |
| `A2_Task_1_and_2_Ryan_Teo.jl` | `src/training.py` (Task 2) | **Task 2**: Baseline classification (DT, SVM, KNN) on all feature schemes. |
| `A2_StaceyFu_Task3_and_4.jl` | `src/training.py` (Task 3/4) | **Task 3 & 4 (Stacey)**: MI Feature Selection + Elastic Net & Voting Ensemble. |
| `A2-Bakhombisile_Dlamini...jl` | `src/training.py` (Task 3/4) | **Task 3 & 4 (Siya)**: mRMR Feature Selection + Nested CV + Gender-specific models. |

### Key Changes
1.  **No more Notebooks**: We use a single script `assignment.py` to run everything.
2.  **Modular Code**:
    *   `src/data.py`: Handles loading the CSV and splitting feature schemes (Baseline, MFCC, etc.).
    *   `src/visualization.py`: Handles all plotting (Task 1).
    *   `src/training.py`: Handles all model training and cross-validation (Tasks 2, 3, 4).
3.  **Results Folder**: All plots and CSVs are automatically saved to `results/` organized by task.

## 🚀 How to Run

You don't need to open multiple files anymore. Just run the main script.

### 1. Setup Environment
First, install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
To run **everything** (Task 1 Visualizations, Task 2 Baselines, Task 3/4 Stacey's Flow, and Task 3/4 Siya's Flow):

```bash
python3 assignment.py --flow both
```

*Note: On macOS, this script automatically handles the OpenBLAS/Matplotlib crash issues we were seeing.*

### 3. Run Specific Parts
If you only want to check your specific sections:

**For Stacey:**
```bash
python3 assignment.py --flow stacey
```
*Runs Task 1, Task 2, then Stacey's Task 3 (Elastic Net) and Task 4 (Ensemble).*

**For Siya:**
```bash
python3 assignment.py --flow siya
```
*Runs Task 1, Task 2, then Siya's Task 3 (Nested CV) and Task 4 (Gender Models).*

## 📂 Output Structure

After running the script, check the `results/` folder. It is organized by task:

```text
results/
├── task1/
│   └── separability_comparison.png      # Fisher Ratio Bar Chart
├── task2/
│   ├── task2_performance.png            # Model Comparison Plot
│   └── task2_results.csv                # Raw Accuracy/F1 scores
├── task3_stacey/
│   ├── stacey_task3_results.csv
│   └── Task_3_-_LogisticRegression...   # PR Curves
├── task4_stacey/
│   └── stacey_task4_results.csv
├── task3_siya/
│   └── ...                              # Nested CV Results & Plots
└── task4_siya/
    └── ...                              # Gender Specific Results & Plots
```

## 🛠️ Developer Notes

*   **Tests**: We added unit tests! Run `pytest` to check if the feature selection and evaluation logic is working correctly.
*   **Refactoring**: `src/modeling.py` was getting too big, so we split it into `src/training.py` (models) and `src/feature_selection.py` (mRMR).

## 🐛 Troubleshooting

*   **"Bus Error" or "Segmentation Fault" on Mac**:
    We have added fixes for this in the code. If it persists, try running with:
    ```bash
    OMP_NUM_THREADS=1 python3 assignment.py --flow both
    ```
