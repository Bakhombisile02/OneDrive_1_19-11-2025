# Comparative Update Plan

## Why This Plan Matters

- Shared infrastructure unlocks direct comparability—common folds, feature views, and reporting remove the "apples vs. oranges" critiques that currently weaken our narrative.
- Feature-scheme benchmarking plus runtime profiling gives us defensible answers to: *Which acoustic families matter most?* *Why is Stacey faster?* *Is Siya’s extra complexity justified?*
- Dual significance testing (bootstrap vs. parametric) lets us argue statistical confidence in whichever method wins, even if distributional assumptions are shaky.
- Regression tests and documentation upgrades ensure the new workflow stays maintainable and reproducible for the upcoming conference demo and our thesis examiners.

## High-Level Objectives
1. **Unified Experiment Harness** – Share folds, feature matrices, logging schema, and runtime tracking across both flows so comparisons are apples-to-apples.
2. **Feature-Scheme Justification** – Quantify how every scheme, fused combo, and the "all features" stack behaves (separability + predictive power) and document why certain schemes are preferred.
3. **Runtime & Complexity Transparency** – Capture wall-clock timings, hardware context, and rough complexity metrics to highlight Stacey’s intentional speed trade-offs.
4. **Dual Significance Analysis** – Provide both non-parametric bootstrap tests and parametric approximations (assuming vs. not assuming a data distribution) for metrics like MCC/AUPRC.
5. **Regression Guardrails** – Add tests ensuring the shared infrastructure behaves deterministically (fold reuse, feature stacking, stats engine correctness).

## Detailed Work Plan

### 1. Shared Evaluation Controller (src/training.py)
- [ ] Implement `run_shared_comparison()` that:
  - Generates a single set of `StratifiedGroupKFold` splits via `get_stratified_group_folds` and persists fold indices to `results/comparison/folds.npy`.
  - Packages standardized inputs: raw `schemes`, stacked matrices (per scheme and fused), labels, subject IDs.
  - Records start/stop times (using `time.perf_counter`) for each pipeline and writes summaries (model, feature view, runtime seconds, hardware info) to `results/comparison/runtime_log.csv`.
  - Orchestrates both flows by injecting the shared folds and collecting per-fold predictions, probabilities, and metadata into JSONL or Parquet artifacts.
  - _Reason: one controller guarantees Stacey and Ryan see exactly what Siya’s nested CV sees, so comparisons stay defensible._

### 2. Stacey Flow Enhancements (src/training.py)
- [ ] Update `run_task_3_stacey`/`run_task_4_stacey` signatures to accept:
  - `fold_indices`, `feature_view` (e.g., per-scheme matrices or fused view), and optional `scheme_name` for logging.
  - Precomputed MI masks or SelectKBest transformations to avoid re-fitting when the shared controller already selected features.
- [ ] Mirror Siya’s metrics by storing MCC, AUPRC, PR curves, and thresholds per fold inside Stacey’s results, ensuring identical schema.
- [ ] Emit runtime + complexity annotations (e.g., number of selected features, tree depth, pipeline steps) into the shared runtime log.
- [ ] Maintain Stacey’s non-nested CV to honor her speed choice; document that nested CV is intentionally skipped but validated through shared folds.
  - _Reason: we respect Stacey’s faster workflow while still benchmarking it under the same folds, keeping her story authentic._

### 3. Siya Flow Alignment (src/training.py)
- [ ] Allow `run_task_3_siya` and `run_task_4_siya` to optionally consume externally supplied folds and feature matrices while preserving the default nested CV for inner-loop tuning.
- [ ] Ensure per-fold outputs follow the same structured schema as Stacey’s for downstream statistical testing (keys: fold, scheme, model, predictions, probabilities, runtime, MCC/AUPRC).
- [ ] When folds are injected, bypass redundant subject splits and simply respect the provided indices to guarantee direct comparability.
  - _Reason: letting Siya reuse shared folds keeps his deeper search fair without forcing extra runtime._

### 4. Feature-Scheme Benchmark Expansion (src/visualization.py & run_task_2)
- [ ] Extend `get_feature_schemes` consumers to generate:
  - Individual schemes (Baseline, Intensity/Formant/Bandwidth, VocalFold, MFCC, Wavelet, TQWT).
  - Fused combos (e.g., MFCC+TQWT, VocalFold+Intensity) and the "All Features" stack.
- [ ] For Task 1 visualizations:
  - Automate Fisher ratio ranking, PCA scatter, and Isomap/t-SNE plots for each scheme + combos; save to `results/task1/{scheme}_*.png`.
  - Produce a consolidated table summarizing separability metrics with confidence intervals.
- [ ] For Task 2 classifiers:
  - Run the standard model suite per scheme/combo/all-features, capturing macro-F1, MCC, recall, AUPRC, plus bootstrap confidence intervals.
  - Generate `results/task2/feature_scheme_comparison.csv` and companion plots for thesis documentation.
  - _Reason: Ryan gets clearer evidence on why certain feature sets shine, which feeds directly into the conference narrative._

### 5. Dual Significance Module (src/evaluation.py)
- [ ] Add utilities:
  - `bootstrap_metric_diff(preds_a, preds_b, metric, n_boot=1000)` returning CI and p-value without distribution assumptions.
  - Parametric option (e.g., DeLong for AUC or paired t-test on MCC logits) gated behind a flag that first attempts to fit a distribution to residuals (Shapiro-Wilk or Anderson-Darling test) to justify the assumption.
  - Aggregated report builder that outputs both results ("Assumed distribution" vs "Non-parametric") for every scheme/model pairing.
- [ ] Wire these into the shared controller post-processing so each Stacey vs Siya comparison automatically emits significance summaries (Markdown + CSV) into `results/comparison/significance/`.
  - _Reason: we can answer tough reviewer questions about "is the difference real?" with both bootstrap and parametric evidence._

### 6. Runtime & Complexity Reporting (new helper in src/training.py or utils)
- [ ] Capture wall-clock time, CPU/GPU availability, and estimated algorithmic complexity markers (e.g., number of parameters, depth, sample size) per run.
- [ ] Provide a lightweight CLI flag (`--profile-comparison`) that toggles deeper profiling (cProfile or sklearn’s `plot_importance` for GBDT) when needed.
- [ ] Surface runtime deltas in the final report so Stacey’s speed trade-off is quantitatively supported.
  - _Reason: highlighting runtime and complexity makes it obvious why Stacey’s streamlined stack is valuable even if accuracy gaps are narrow._

### 7. Testing & Validation (tests/)
- [ ] Add tests for:
  - Shared fold generator returning identical assignments when invoked twice with same seed and subject IDs.
  - Feature stacking logic ensuring fused matrices contain expected column counts/order.
  - Significance module detecting differences on synthetic datasets (e.g., two classifiers with known AUPRC gap) for both bootstrap and parametric paths.
  - Runtime logger producing monotonic timestamps and non-empty hardware metadata.
- [ ] Update any existing smoke tests (`run_smoke_test.sh`) to hit the new shared comparison path with reduced folds for CI feasibility.

### 8. Documentation & Reporting
- [ ] Update `README.md` with new CLI flags (e.g., `--comparison-mode`, `--profile-comparison`) and describe how to interpret runtime + significance outputs.
- [ ] Add a thesis-facing appendix summarizing methodology alignment, scheme evaluations, and statistical findings (possibly in `docs/thesis_appendix.md`).
- [ ] Ensure plots/tables reference the new shared artifacts so narratives about "why Stacey vs Siya" are backed by reproducible evidence.

## Assumptions & Open Questions
1. **Runtime Capture Scope** – Are we logging just total Task 3/4 durations, or also per-fold/per-model times? Current plan assumes both for fidelity.
2. **Distribution Fit Decision** – Default to Shapiro-Wilk at α=0.05 for deciding whether parametric tests are defensible; fall back to bootstrap otherwise, but still report the parametric attempt for completeness.
3. **Hardware Variability** – Runtime comparisons should record interpreter, CPU, GPU availability, and threads used to contextualize results (important if Siya leverages GPU-only options).

This plan keeps Stacey’s and Siya’s methodological identities intact while layering the infrastructure needed to argue—quantitatively and statistically—whose approach prevails and under what conditions.
