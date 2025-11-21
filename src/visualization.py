import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from src.config import SEED
from src.data import get_feature_schemes, standardize

def visualize_scheme(X, y, scheme_name, save_plot=False):
    """
    Visualize the feature scheme using PCA, Isomap, and t-SNE.
    """
    print(f"Visualizing {scheme_name}...")
    
    # PCA reduction first if dims > 50
    n_samples, n_features = X.shape
    pca_dim = min(50, n_features, n_samples - 1)
    
    pca = PCA(n_components=pca_dim, random_state=SEED)
    X_pca = pca.fit_transform(X)
    
    # Isomap
    iso = Isomap(n_neighbors=15, n_components=2)
    X_iso = iso.fit_transform(X_pca)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)
    
    # Plotting
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
        plt.savefig(f"viz_{scheme_name}.png")
    # plt.show() # Avoid blocking

def compute_fisher_ratio(X, y):
    """
    Compute the Fisher Ratio for class separability.
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
    
    # Between-class scatter
    Sb = n_hc * np.outer(mu_hc - mu_all, mu_hc - mu_all) + \
         n_pd * np.outer(mu_pd - mu_all, mu_pd - mu_all)
         
    # Within-class scatter
    # Covariance in numpy is normalized by N-1 by default (ddof=1)
    Sw_hc = np.cov(X_hc, rowvar=False, ddof=1)
    Sw_pd = np.cov(X_pd, rowvar=False, ddof=1)
    
    # Handle scalar covariance for 1D features (though unlikely here)
    if Sw_hc.ndim == 0:
        Sw_hc = np.array([[Sw_hc]])
    if Sw_pd.ndim == 0:
        Sw_pd = np.array([[Sw_pd]])
    
    Sw = (n_hc * Sw_hc + n_pd * Sw_pd) / (n_hc + n_pd)
    
    # Fisher Ratio = tr(Sb) / tr(Sw)
    # Add small epsilon to trace of Sw to avoid division by zero
    fisher_ratio = np.trace(Sb) / (np.trace(Sw) + 1e-9)
    
    return fisher_ratio

def run_task_1(df):
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
    # for name, X in standardized_schemes.items():
    #     visualize_scheme(X, y, name, save_plot=True)
        
    # Separability
    results = []
    for name, X in standardized_schemes.items():
        fr = compute_fisher_ratio(X, y)
        results.append({'Scheme': name, 'N_Features': X.shape[1], 'Fisher_Ratio': fr})
        
    results_df = pd.DataFrame(results).sort_values(by='Fisher_Ratio', ascending=False)
    print("\nSeparability Ranking:")
    print(results_df)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Scheme', y='Fisher_Ratio', data=results_df, palette='viridis')
    plt.title("PD vs HC Separability by Feature Scheme")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("separability_comparison.png")
    # plt.show() # Commented out to avoid blocking execution
    
    # Return RAW schemes to prevent leakage
    return schemes, y
