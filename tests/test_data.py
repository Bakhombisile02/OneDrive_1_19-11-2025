import numpy as np
import pandas as pd

from src.data import get_feature_schemes


def _mock_dataframe(rows=12):
    cols = [f"col_{i}" for i in range(755)]
    data = np.random.rand(rows, 755)
    return pd.DataFrame(data, columns=cols)


def test_get_feature_schemes_includes_all_features_and_fusions():
    df = _mock_dataframe()
    schemes = get_feature_schemes(df)

    assert "AllFeatures" in schemes
    assert "MFCC_TQWT_Fusion" in schemes
    assert "Intensity_Vocal_Fusion" in schemes

    all_shape = schemes["AllFeatures"].shape[1]
    # Expect columns 2 through 753 (exclusive of 754 label)
    assert all_shape == 752

    fusion_shape = schemes["MFCC"].shape[1] + schemes["TQWT"].shape[1]
    assert schemes["MFCC_TQWT_Fusion"].shape[1] == fusion_shape
