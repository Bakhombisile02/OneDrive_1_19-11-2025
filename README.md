# INFO411 Assignment 2: Parkinson's Disease Detection

This project implements a machine learning pipeline for detecting Parkinson's Disease from speech features. It includes data visualization, baseline modeling, feature selection (mRMR), and advanced nested cross-validation with ensemble methods.

## Project Structure

- `src/`: Modular source code (data loading, visualization, modeling, evaluation).
- `data/`: Dataset storage (downloaded automatically).
- `legacy/`: Original Julia scripts and older files.
- `assignment.py`: Main entry point for running the analysis.
- `setup.sh`: Script to set up the environment from scratch.

## Quick Start

For new machines, run the setup script to create a virtual environment, install dependencies, and download the data:

```bash
./setup.sh
```

Then, activate the environment:

```bash
source .venv/bin/activate
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

# Run only Siya's pipeline (Task 3b/4b)
python assignment.py --flow siya
```

## Options

- `--siya-device [auto|cpu]`: Force CPU usage or allow auto-detection of GPU (CUDA/Metal).
- `--siya-enhanced`: Enable broader hyperparameter grids and calibration.
- `--siya-parallel-configs`: Parallelize inner CV loops (memory intensive).

## Testing

Run the smoke test to verify the installation:

```bash
./run_smoke_test.sh
```
