import pandas as pd
from src.features.features_config import get_feature_config

def compute_ma(series, window):
    return series.rolling(window).mean().round(2)

def compute_diff(series, lag):
    return series.diff(lag).round(2)

def compute_gap(series, window):
    return series - series.rolling(window).mean().round(2)


def apply_transformations(df_feature, feature_name, config=None):

    if config is None:
        config = get_feature_config()[feature_name]

    ops = config.get("transformations", {})
    df_out = df_feature.copy()

    for op, params in ops.items():
        if op == "ma":
            for w in params["windows"]:
                df_out[f"{feature_name}_MA{w}"] = compute_ma(df_out[feature_name], w)
        elif op == "diff":
            for lag in params["lags"]:
                df_out[f"{feature_name}_DIFF{lag}"] = compute_diff(df_out[feature_name], lag)
        elif op == "gap":
            for w in params["windows"]:
                df_out[f"{feature_name}_GAP{w}"] = compute_gap(df_out[feature_name], w)

    return df_out
