import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fix for macOS bus error/segfault
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from src.config import SEED
from src.data import get_feature_schemes, standardize

import os

def visualize_scheme(X, y, scheme_name, save_plot=False, output_dir=".", top_k=6):
    """Generate comparative PCA + feature effect visualisations for a scheme."""
    print(f"Visualizing {scheme_name}...")
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)
    feature_names = [str(col) for col in X_df.columns]
    X_df.columns = feature_names
    y_series = pd.Series(y).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    n_samples, n_features = X_df.shape
    if n_samples < 3 or n_features == 0:
        print("  Skipping visualization (insufficient samples/features).")
        return {}

    palette = {0: 'steelblue', 1: 'firebrick'}
    label_map = {0: 'HC', 1: 'PD'}

    pca_components = min(2, n_features, n_samples - 1)
    if pca_components < 1:
        pca_components = 1
    pca = PCA(n_components=pca_components, random_state=SEED)
    X_pca = pca.fit_transform(X_df.values)
    explained = pca.explained_variance_ratio_
    if pca_components == 1:
        X_pca = np.hstack([X_pca, np.zeros_like(X_pca)])

    group_means = X_df.groupby(y_series).mean()
    diff = (group_means.loc[1] - group_means.loc[0]).abs()
    std = X_df.std(ddof=1) + 1e-9
    effect = (diff / std).fillna(0)
    effect = effect.sort_values(ascending=False)
    top_features = effect.head(min(top_k, len(effect)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PCA Scatter
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=y_series.map(label_map),
        palette=[palette[0], palette[1]],
        alpha=0.65,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(f"{scheme_name}: PCA Space")
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].legend(title='Class')

    # Explained variance bar
    ev = list(explained) + [0.0] * (2 - len(explained))
    axes[0, 1].bar(['PC1', 'PC2'], ev, color=['#4c72b0', '#dd8452'])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel('Explained Variance')
    axes[0, 1].set_title('Variance Captured')

    # Effect size bar chart
    sns.barplot(
        x=top_features.values,
        y=top_features.index,
        palette='mako',
        ax=axes[1, 0],
    )
    axes[1, 0].set_title('Top Feature Effect Sizes (|Δμ|/σ)')
    axes[1, 0].set_xlabel('Effect Size')
    axes[1, 0].set_ylabel('Feature')

    # Distribution plot for top 2 features
    dist_features = top_features.index[: min(2, len(top_features))]
    if len(dist_features) == 0:
        axes[1, 1].axis('off')
    else:
        melted = X_df[dist_features].copy()
        melted['Class'] = y_series.map(label_map)
        melted = melted.melt(id_vars='Class', var_name='Feature', value_name='Value')
        sns.violinplot(
            data=melted,
            x='Value',
            y='Feature',
            hue='Class',
            split=True,
            palette=[palette[0], palette[1]],
            ax=axes[1, 1],
        )
        axes[1, 1].set_title('Class Distributions (Top Features)')
        axes[1, 1].legend(loc='best')

    plt.suptitle(f"{scheme_name}: Comparative View", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_plot:
        save_path = os.path.join(output_dir, f"viz_{scheme_name}.png")
        print(f"  Saving plot to {save_path}...")
        plt.savefig(save_path)
    plt.close(fig)

    summary = {
        'ExplainedVar_PC1': float(ev[0]),
        'ExplainedVar_PC2': float(ev[1]),
        'MeanEffect': float(effect.mean()),
        'TopFeatureEffect': float(top_features.iloc[0]) if len(top_features) else 0.0,
        'TopFeatureName': str(top_features.index[0]) if len(top_features) else None,
    }
    return summary

def compute_fisher_ratio(X, y):
    """
    Compute the Fisher Ratio for class separability.
    Optimized to avoid full covariance matrix computation (trace only).
    """
    X = np.array(X)
    y = np.array(y)
    
    X_hc = X[y == 0]
    X_pd = X[y == 1]
    
    mu_hc = np.mean(X_hc, axis=0)
    mu_pd = np.mean(X_pd, axis=0)
    mu_all = np.mean(X, axis=0)
    
    n_hc = X_hc.shape[0]
    n_pd = X_pd.shape[0]
    
    # Trace of Between-class scatter (Sb)
    # tr(Sb) = n_hc * ||mu_hc - mu_all||^2 + n_pd * ||mu_pd - mu_all||^2
    tr_Sb = n_hc * np.sum((mu_hc - mu_all)**2) + \
            n_pd * np.sum((mu_pd - mu_all)**2)
         
    # Trace of Within-class scatter (Sw)
    # tr(Sw) = (n_hc * tr(Sw_hc) + n_pd * tr(Sw_pd)) / (n_hc + n_pd)
    # tr(Sw_class) = sum of variances of features in that class
    var_hc = np.var(X_hc, axis=0, ddof=1)
    var_pd = np.var(X_pd, axis=0, ddof=1)
    
    tr_Sw_hc = np.sum(var_hc)
    tr_Sw_pd = np.sum(var_pd)
    
    tr_Sw = (n_hc * tr_Sw_hc + n_pd * tr_Sw_pd) / (n_hc + n_pd)
    
    # Fisher Ratio = tr(Sb) / tr(Sw)
    fisher_ratio = tr_Sb / (tr_Sw + 1e-9)
    
    return fisher_ratio

def run_task_1(df, output_dir="."):
    """
    Run Task 1: Visualization and Separability Analysis.
    
    Returns:
        schemes (dict): Dictionary of RAW feature schemes (not standardized).
        y (pd.Series): Target labels.
    """
    print("\n--- Task 1: Visualization & Separability ---")
    # Class label (column 755 in Julia -> 754 in Python 0-indexed)
    y = df.iloc[:, 754].astype(int)
    
    schemes = get_feature_schemes(df)
    
    # Standardize ONLY for visualization and Fisher Ratio calculation
    # We do NOT return these standardized schemes to avoid data leakage in later tasks
    standardized_schemes = {name: standardize(X) for name, X in schemes.items()}
    
    print("Computing scheme summaries...")
    results = []
    for name, X in standardized_schemes.items():
        summary = visualize_scheme(X, y, name, save_plot=True, output_dir=output_dir)
        print(f"  Computing Fisher Ratio for {name}...")
        fr = compute_fisher_ratio(X, y)
        entry = {
            'Scheme': name,
            'N_Features': X.shape[1],
            'Fisher_Ratio': fr,
        }
        if summary:
            entry.update(summary)
        results.append(entry)
        
    results_df = pd.DataFrame(results).sort_values(by='Fisher_Ratio', ascending=False)
    print("\nSeparability Ranking:")
    print(results_df)
    
    print("Plotting Fisher Ratios...")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Scheme', y='Fisher_Ratio', data=results_df, palette='viridis')
    plt.title("PD vs HC Separability by Feature Scheme")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "separability_comparison.png")
    plt.savefig(save_path)
    plt.close() # Close to free memory
    print(f"Saved separability_comparison.png to {save_path}")

    metric_heatmap_path = os.path.join(output_dir, "scheme_metric_heatmap.png")
    plot_scheme_metric_heatmap(results_df, metric_heatmap_path)
    
    # Return RAW schemes to prevent leakage
    return schemes, y

def plot_task2_performance(results_df, save_path="results/task2_performance.png"):
    """
    Plot the performance of different models across feature schemes (Task 2).
    """
    if results_df.empty:
        print("No results to plot for Task 2.")
        return

    plt.figure(figsize=(14, 7))
    sns.barplot(
        x='Scheme', y='Accuracy_mean', hue='Model', data=results_df,
        palette='viridis'
    )
    plt.title("Task 2: Classification Accuracy by Feature Scheme")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Feature Scheme")
    plt.xticks(rotation=45)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved Task 2 plot to {save_path}")


def plot_scheme_metric_heatmap(metrics_df, save_path):
    if metrics_df.empty:
        return
    metric_cols = [
        'Fisher_Ratio',
        'ExplainedVar_PC1',
        'ExplainedVar_PC2',
        'MeanEffect',
        'TopFeatureEffect',
    ]
    available = [c for c in metric_cols if c in metrics_df.columns]
    if not available:
        return

    matrix = metrics_df.set_index('Scheme')[available]
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='crest',
        cbar_kws={'label': 'Relative Score'},
    )
    plt.title('Feature Scheme Comparative Metrics')
    plt.ylabel('Scheme')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved scheme metric heatmap to {save_path}")


def plot_pr_curves(results_list, title_prefix="Task 3", save_dir="results"):
    """
    Plot Precision-Recall curves for nested CV results.
    """
    if not results_list:
        return

    plt.figure(figsize=(10, 6))
    
    for res in results_list:
        fold = res.get('Fold', 'Unknown')
        recall, precision = res.get('PR_Curve', ([], []))
        auprc = res.get('AUPRC', 0)
        
        plt.plot(
            recall, precision, 
            label=f'Fold {fold} (AUPRC = {auprc:.3f})'
        )
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title_prefix}: Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    filename = f"{save_dir}/{title_prefix.replace(' ', '_')}_PR_Curve.png"
    plt.savefig(filename)
    print(f"Saved PR Curve to {filename}")
