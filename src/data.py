import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """
    Load the dataset from the specified path.
    """
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape}")
    return df

def get_feature_schemes(df):
    """
    Extract feature schemes from the dataframe.
    
    Returns a dictionary of DataFrames for each scheme.
    """
    # Based on Julia code indices (1-based in Julia, 0-based in Python)
    # Julia: 3:23 -> Python: 2:23 (iloc)
    # Julia: 24:34 -> Python: 23:34
    # ...
    # Note: Julia ranges are inclusive-inclusive, Python is inclusive-exclusive
    
    schemes = {
        "Baseline": df.iloc[:, 2:23],
        "IntensityFormantBandwidth": df.iloc[:, 23:34],
        "VocalFold": df.iloc[:, 34:56],
        "MFCC": df.iloc[:, 56:140],
        "Wavelet": df.iloc[:, 140:322],
        "TQWT": df.iloc[:, 322:754]
    }

    def _fuse(name, frames):
        fused = np.hstack([f.values for f in frames])
        n_cols = fused.shape[1]
        columns = [f"{name}_f{i}" for i in range(n_cols)]
        return pd.DataFrame(fused, columns=columns)

    # Fusion views demanded by comparative analysis
    schemes["MFCC_TQWT_Fusion"] = _fuse(
        "mfcc_tqwt", [schemes["MFCC"], schemes["TQWT"]]
    )
    schemes["Intensity_Vocal_Fusion"] = _fuse(
        "intensity_vocal", [
            schemes["IntensityFormantBandwidth"],
            schemes["VocalFold"],
        ]
    )

    # All-features view keeps ordering consistent with downstream concatenations
    feature_block = df.iloc[:, 2:754]
    feature_block.columns = [f"feat_{i}" for i in range(feature_block.shape[1])]
    schemes["AllFeatures"] = feature_block

    return schemes

def standardize(X):
    """
    Standardize the data (Zero mean, Unit variance).
    Useful for visualization, but avoid using on full dataset before CV to prevent leakage.
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns if isinstance(X, pd.DataFrame) else None)
