from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# ATC_CATEGORIES = ("M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06")
# IF_FEATURES = (
#     "qty_lag_origin", "qty_lag_origin_7", "qty_roll_mean_7",
#     "qty_roll_std_7", "qty_roll_mean_28", "qty_roll_std_28",
#     "qty_mean_ratio_7_28", "qty_last_zscore_28", "qty_zero_ratio_28",
#     "weekday", "annual_sin", "annual_cos",
# )
# TARGET_COL = "quantity"
# NON_PREDICTOR_COLUMNS = {
#     "date", "category", "forecast_origin", "horizon_days", TARGET_COL, "split",
# }

DATE_COL = "date" #ou datum dependendo qual notebook esta
TARGET_COL = "quantity"

# These fields are known before the beginning of the forecast date. All
# rolling variables are created with shift(1) in DataQualityFeatures.
LEAKAGE_SAFE_NUMERIC_FEATURES = [
    "qty_lag_origin", 
    "qty_lag_origin_7", 
    "qty_roll_mean_7",
    "qty_roll_std_7", 
    "qty_roll_mean_28", 
    "qty_roll_std_28",
    "qty_mean_ratio_7_28", 
    "qty_last_zscore_28", 
    "qty_zero_ratio_28",
    "weekday", 
    "annual_sin", 
    "annual_cos",
]

#LEAKAGE_SAFE_CATEGORICAL_FEATURES = ["Product", "Region"] nao tem

LAGGED_ANOMALY_FEATURES = [
    "anomaly_score",
    "is_anomaly"
]

LEAKAGE_SAFE_HYBRID_NUMERIC_FEATURES = (
    LEAKAGE_SAFE_NUMERIC_FEATURES + LAGGED_ANOMALY_FEATURES
)


def sort_by_forecast_date(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """Return a date-sorted dataframe with a clean positional index."""
    if date_col not in df.columns:
        raise ValueError(f"Missing temporal validation column: {date_col}")

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="raise")
    return result.sort_values(date_col).reset_index(drop=True)

def build_date_based_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    date_col: str = DATE_COL,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window folds without splitting the same date.

    ``TimeSeriesSplit`` is applied to unique dates rather than dataframe rows.
    This is important for panel data because all Product/Region rows belonging
    to one day must stay on the same side of a fold.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing temporal validation column: {date_col}")

    dates = pd.to_datetime(df[date_col], errors="raise")
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Dataframe must be sorted by forecast date before building CV splits."
        )

    unique_dates = pd.Index(dates.drop_duplicates())

    if len(unique_dates) <= n_splits:
        raise ValueError(
            f"At least {n_splits + 1} unique dates are required; "
            f"received {len(unique_dates)}."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_date_positions, validation_date_positions in splitter.split(unique_dates):
        train_dates = unique_dates[train_date_positions]
        validation_dates = unique_dates[validation_date_positions]

        train_rows = np.flatnonzero(dates.isin(train_dates).to_numpy())
        validation_rows = np.flatnonzero(dates.isin(validation_dates).to_numpy())

        if dates.iloc[train_rows].max() >= dates.iloc[validation_rows].min():
            raise RuntimeError("Invalid temporal split: training reaches validation dates.")

        splits.append((train_rows, validation_rows))

    return splits

def calendar_diagnostics(df: pd.DataFrame, date_col: str,freq: str = "D") -> Dict[str, Any]:
        """Verifica lacunas no calendário ANTES de decidir como tratá-las.

        Não modifica o DataFrame. Datas duplicadas continuam sendo erro, pois
        uma instância do AutoETS deve representar uma única série ATC diária.
        """
        if date_col not in df.columns:
            raise ValueError(f"Coluna de data ausente: {date_col!r}.")
        if df.empty:
            raise ValueError("DataFrame vazio.")
        dates = pd.to_datetime(df[date_col], errors="raise").dt.normalize()
        if dates.isna().any():
            raise ValueError("Datas nulas.")
        unique = pd.DatetimeIndex(dates).unique().sort_values()
        complete = pd.date_range(unique[0], unique[-1], freq=freq)
        gaps = complete.difference(unique)
        duplicates = len(dates) - len(unique)
        return {
            "date_start": str(unique[0].date()),
            "date_end": str(unique[-1].date()),
            "existing_rows": len(df),
            "expected_days": len(complete),
            "duplicate_dates": int(duplicates),
            "missing_days": len(gaps),
            "missing_pct": float(100 * len(gaps) / len(complete)),
            "missing_dates_sample": [str(ts.date()) for ts in gaps[:10]],
        }

