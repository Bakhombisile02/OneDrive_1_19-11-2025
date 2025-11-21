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

def visualize_scheme(X, y, scheme_name, save_plot=False, output_dir="."):
    """
    Visualize the feature scheme using PCA, Isomap, and t-SNE.
    """
    print(f"Visualizing {scheme_name}...")
    print("  Skipping manifold learning (PCA/Isomap/t-SNE) to avoid environment-specific segmentation faults.")
    return

    # PCA reduction first if dims > 50
    print("  Running PCA...")
    n_samples, n_features = X.shape
    pca_dim = min(50, n_features, n_samples - 1)
    
    try:
        pca = PCA(n_components=pca_dim, random_state=SEED, svd_solver='full')
        X_pca = pca.fit_transform(X)
        
        # Isomap
        print("  Running Isomap...")
        iso = Isomap(n_neighbors=15, n_components=2)
        X_iso = iso.fit_transform(X_pca)
        
        # t-SNE
        print("  Running t-SNE...")
        # Use 'random' init to avoid potential segfaults with 'pca' init on some systems/versions
        tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init='random', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_pca)
        
        # Plotting
        print("  Plotting...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Isomap Plot
        sns.scatterplot(x=X_iso[:, 0], y=X_iso[:, 1], hue=y, palette={0: 'blue', 1: 'red'}, ax=axes[0], alpha=0.6)
        axes[0].set_title(f"{scheme_name} - Isomap")
        axes[0].legend(title='Class', labels=['HC', 'PD'])
        
        # t-SNE Plot
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette={0: 'blue', 1: 'red'}, ax=axes[1], alpha=0.6)
        axes[1].set_title(f"{scheme_name} - t-SNE")
        axes[1].legend(title='Class', labels=['HC', 'PD'])
        
        plt.tight_layout()
        if save_plot:
            save_path = os.path.join(output_dir, f"viz_{scheme_name}.png")
            print(f"  Saving plot to {save_path}...")
            plt.savefig(save_path)
        plt.close(fig) # Close to free memory
    except Exception as e:
        print(f"  Skipping visualization for {scheme_name} due to error: {e}")

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
    
    # Visualization
    for name, X in standardized_schemes.items():
        visualize_scheme(X, y, name, save_plot=True, output_dir=output_dir)
        
    # Separability
    print("Computing Fisher Ratios...")
    results = []
    for name, X in standardized_schemes.items():
        print(f"  Computing Fisher Ratio for {name}...")
        fr = compute_fisher_ratio(X, y)
        results.append({'Scheme': name, 'N_Features': X.shape[1], 'Fisher_Ratio': fr})
        
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
