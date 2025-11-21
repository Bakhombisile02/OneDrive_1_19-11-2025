# Copilot Instructions
## Big Picture
- INFO411 Assignment 2 targets PD screening from sustained /a/ phonations; Tasks 1-4 cover visualization, baseline modelling, feature selection, and focused improvements, and the write-up mirrors these sections verbatim.
- `assignment.py` is the entry point and chains Tasks 1-4 at the bottom; each `run_task_*` consumes the previous task's outputs, so keep return signatures stable when refactoring.
- Every task operates on the 755-column `Parkinsons_Speech-Features.csv` file: column 0 is `subject_id`, column 1 encodes gender, columns 2-753 are feature blocks, and column 754 is the `class` label (0=HC, 1=PD).
- Feature engineering revolves around `get_feature_schemes`, which slices the master DataFrame into Baseline, Intensity/Formant/Bandwidth, VocalFold, MFCC, Wavelet, TQWT, and a fused MFCC+TQWT view; downstream code expects the dictionary keys to stay consistent.
- Julia Pluto notebooks (`A2_*.jl`) mirror experimentation from earlier milestones but are not invoked by the Python workflow; treat them as methodological references only (e.g., Stacey’s MI+Elastic Net cells, Siya’s nested CV logs).

## Data & Feature Handling
- Dataset recap: 252 subjects × 3 recordings = 756 rows, PD prevalence ≈74.6%; columns 2-754 chunk into contiguous feature families documented in the report (Baseline 2:23, Intensity/Formant/Bandwidth 23:34, VocalFold 34:56, MFCC 56:140, Wavelet 140:322, TQWT 322:754, label 754).
- Always call `standardize` before dimensionality reduction or modeling so that each scheme stays zero-mean/unit-variance with aligned column names; Task 3/4 re-fit scalers inside each fold to stay leakage-safe.
- If you add a new scheme, keep indices 0-based and update any resulting concatenations (`pd.concat` / `np.hstack`) so row order remains synchronized across schemes before `subject_id` grouping.
- `run_task_1` uses PCA→Isomap/t-SNE for visualization (optionally writing `viz_{scheme}.png`) and ranks separability via `compute_fisher_ratio`; the plotted summary is saved to `separability_comparison.png` and backs the report’s Figure 1 and Table 1.
- `schemes` are passed around as Pandas DataFrames; Task 3/4 convert them to `np.hstack` arrays, so avoid introducing ragged structures or missing values.

## Modeling & Evaluation Workflow
- `create_subject_folds` enforces grouped K-folds by subject; reuse it (or extend it) whenever you introduce new CV logic to avoid leakage between recordings of the same subject.
- Task 1’s manifolds and Fisher ratios provide the qualitative/quantitative separability narrative used in Section 3 of the report; if you change the visualization stack, update the write-up references.
- Task 2 iterates over each scheme with Decision Trees, SVMs, and KNNs using manual folds; metrics are aggregated into a DataFrame, so new models should follow the same "per fold metrics → avg/std" pattern before updating Table 2.
- Task 3 (`nested_cv_run`) performs mRMR feature selection (`mrmr_rank` uses joblib + quantile discretization) inside nested CV for SVM/XGBoost/GBDT; if XGBoost fails to import, the helper automatically falls back to `GradientBoostingClassifier` with compatible hyperparameters.
- Task 3 Stacey's path (`run_task_3_stacey`) builds an MI filter + Elastic Net logistic regression pipeline and a balanced Random Forest comparator via scikit-learn `Pipeline`; keep `k` (default 150) synchronized between the selector and downstream models.
- Task 4 Stacey reuses unfitted pipelines inside a `VotingClassifier` (soft voting) to mirror the report’s Section 6.1, while Task 4 Siya calls `nested_cv_run` separately per gender subset with MCC-tuned thresholds; when extending, be mindful of class imbalance (some genders may lack both labels).
- Macro-F1, recall, AUPRC, and MCC are the headline metrics cited throughout the report; use `compute_metrics` plus `precision_recall_curve` to keep the narrative consistent.

## Tooling, Commands, and Dependencies
- Dependencies span pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, kagglehub, and (optionally) xgboost; there is no `requirements.txt`, so document new libraries explicitly in PRs.
- Missing-but-required imports for existing code include `Pipeline`, `SelectKBest`, `mutual_info_classif`, `LogisticRegression`, `RandomForestClassifier`, and `VotingClassifier`; keep related imports consolidated near the top of `assignment.py`.
- Fetch the dataset with `python download_data.py` (requires Kaggle credentials configured for `kagglehub`); the script copies the first CSV it finds to `Parkinsons_Speech-Features.csv` in the repo root.
- Run the full analysis with `python assignment.py`; expect long runtimes because t-SNE and nested CV are expensive—comment out specific `run_task_*` calls when iterating on one stage.
- Device-aware training lives in `get_xgboost_device_params`, which selects CUDA (if `torch` detects a GPU) or tuned CPU settings for Apple Silicon; reuse it for any future gradient-boosting integrations.

## Implementation Patterns & Tips
- Respect the global `SEED` constant so visualizations and CV splits stay reproducible across runs and reports; the write-up assumes deterministic folds for Part 1/2 and logged seeds for Part 3/4.
- All metrics funnel through `compute_metrics`; extend that helper instead of re-implementing Accuracy/Sensitivity/Specificity logic in new experiments.
- When adding plots, prefer the existing seaborn/matplotlib style (legends: HC blue, PD red) and gate `plt.show()` to avoid blocking automated runs.
- Heavy computations (mRMR, nested CV) already leverage `joblib.Parallel`; if you introduce new loops, profile before adding more parallel layers to avoid oversubscribing cores.
- Keep the report’s division of labour (Ryan: Tasks 1-2, Stacey/Bakhombisile: Tasks 3-4) in mind when attributing new results; contributions tables depend on these boundaries.
