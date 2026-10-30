"""PharmaDemand: um DataFrame em memória com features LightGBM + IF.

Fluxo:
    df_salesdaily = pd.read_csv('data/raw/salesdaily.csv')
    df_features, artifacts = build_features_dataframe(df_salesdaily)
    df_features.to_parquet('data/processed/salesdaily_features_N02BE.parquet', index=False)

CLI (na raiz do projeto):
    python -m src.features.build_features --category N02BE --horizons 7 15

A linha (date=t, horizon_days=h) usa apenas histórico até forecast_origin=t-h.
'quantity' é o alvo, nunca preditora. Os indicadores do IF descrevem o
CONTEXTO conhecido em t-h, não uma anomalia da venda futura em t.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.repository.parquet_repository import ParquetRepository
from src.dataProcessing.data_quality import DataQuality


ROOT = Path(__file__).resolve().parents[2]
ATC_CATEGORIES = ("M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06")
IF_FEATURES = (
    "qty_lag_origin", "qty_lag_origin_7", "qty_roll_mean_7",
    "qty_roll_std_7", "qty_roll_mean_28", "qty_roll_std_28",
    "qty_mean_ratio_7_28", "qty_last_zscore_28", "qty_zero_ratio_28",
    "weekday", "annual_sin", "annual_cos",
)
TARGET_COL = "quantity"
NON_PREDICTOR_COLUMNS = {
    "date", "category", "forecast_origin", "horizon_days", TARGET_COL, "split",
}


def load_sales(csv_path: Path, category: str) -> pd.Series:
    """Compatibilidade com código anterior; no pipeline principal o CSV é lido 1 vez."""
    return pd.read_csv(csv_path, usecols=["datum", category]), category


def make_features(series: pd.Series, category: str, horizon: int) -> pd.DataFrame:
    """Gera features por horizonte, com corte estrito no último dia conhecido t-h."""

    if horizon <= 0:
        raise ValueError("horizon deve ser >= 1")
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_monotonic_increasing:
        raise ValueError("series deve usar índice DatetimeIndex crescente")
    if not series.index.is_unique:
        raise ValueError("A série contém datas repetidas")
    
    idx = series.index
    y = series.astype(float)
    df = pd.DataFrame(index=idx)

    df["date"] = idx
    df["category"] = category
    df["horizon_days"] = int(horizon)
    df["forecast_origin"] = idx - pd.Timedelta(days=horizon)
    df[TARGET_COL] = y  # Rótulo disponível somente no treinamento retrospectivo

    # Datas futuras do calendário são conhecidas na origem.
    df["year"] = idx.year
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["weekday"] = idx.dayofweek
    df["dayofyear"] = idx.dayofyear
    df["weekofyear"] = idx.isocalendar().week.to_numpy(dtype=int)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)


    df["weekday_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7) 
    df["weekday_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["annual_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["annual_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    known = y.shift(horizon)
    for back in (0, 1, 7, 14, 28, 56):
        name = "qty_lag_origin" if back == 0 else f"qty_lag_origin_{back}"
        df[name] = known.shift(back)

    for window in (7, 14, 28, 56):
        rolling = known.rolling(window=window, min_periods=window)
        df[f"qty_roll_mean_{window}"] = rolling.mean()
        df[f"qty_roll_std_{window}"] = rolling.std(ddof=0)
        if window in (7, 28):
            df[f"qty_roll_sum_{window}"] = rolling.sum()
            df[f"qty_roll_min_{window}"] = rolling.min()
            df[f"qty_roll_max_{window}"] = rolling.max()
    
    # avoids division by zero; evita divisao por zero
    eps = 1e-6

    mean7 = df["qty_roll_mean_7"]
    mean28 = df["qty_roll_mean_28"]
    std28 = df["qty_roll_std_28"]
    df["qty_mean_ratio_7_28"] = mean7 / (mean28 + eps)
    df["qty_mean_delta_7_28"] = mean7 - mean28
    df["qty_last_ratio_28"] = known / (mean28 + eps)
    df["qty_last_zscore_28"] = (known - mean28) / (std28 + eps)
    df["qty_cv_28"] = std28 / (mean28 + eps)
    df["qty_zero_ratio_28"] = known.eq(0).rolling(28, min_periods=28).mean()
    df["qty_mean_ratio_14_56"] = df["qty_roll_mean_14"] / (
        df["qty_roll_mean_56"] + eps
    )

    df = df.dropna(axis=0).reset_index(drop=True)
    if df.empty:
        raise ValueError("Histórico insuficiente para gerar o horizonte solicitado")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Preditoras LightGBM, incluindo anomaly_score e is_anomaly após o IF."""
    return [c for c in df.columns if c not in NON_PREDICTOR_COLUMNS]


def fit_if_train_only(
    df: pd.DataFrame, cutoff: pd.Timestamp, contamination: float = 0.03,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Ajusta IF APENAS no treino, pontua treino/teste sem usar a venda alvo."""
    if not (0 < contamination < 0.5):
        raise ValueError("contamination precisa estar entre 0 e 0.5")
    out = df.copy()
    train_mask = out["date"] <= cutoff
    if train_mask.sum() < 200 or (~train_mask).sum() < 1:
        raise ValueError("Quantidade insuficiente de dados no treino ou teste")
    for col in IF_FEATURES:
        if col not in out.columns:
            raise KeyError(f"Feature IF não encontrada: {col}")
    X_train = out.loc[train_mask, list(IF_FEATURES)]
    X_all = out.loc[:, list(IF_FEATURES)]
    if not np.isfinite(X_all.to_numpy(dtype=float)).all():
        raise ValueError("Há valores não finitos nas features IF")

    model = IsolationForest(
        n_estimators=250,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    raw_train = -model.score_samples(X_train)  # maior = mais anômalo
    raw_all = -model.score_samples(X_all)
    train_min = float(raw_train.min())
    train_max = float(raw_train.max())
    denom = max(train_max - train_min, 1e-12)
    score_train = np.clip((raw_train - train_min) / denom, 0.0, 1.0)
    score_all = np.clip((raw_all - train_min) / denom, 0.0, 1.0)
    threshold = float(np.quantile(score_train, 1.0 - contamination))

    out["anomaly_score"] = score_all
    out["is_anomaly"] = (out["anomaly_score"] >= threshold).astype(np.int8)
    out["split"] = np.where(train_mask, "train", "test")
    artifact = {
        "model": model,
        "feature_columns": list(IF_FEATURES),
        "score_min_train": train_min,
        "score_max_train": train_max,
        "threshold": threshold,
        "contamination": float(contamination),
        "train_cutoff": cutoff.strftime("%Y-%m-%d"),
        "semantics": "Anomaly of historical context at forecast_origin; NOT target-day anomaly",
    }
    return out, artifact


def build_features_dataframe(
    df_salesdaily: pd.DataFrame,
    category: str = "N02BE",
    horizons: tuple[int, ...] | list[int] = (7, 15),
    test_months: int = 6,
    contamination: float = 0.03,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Recebe dados crus e retorna UM DataFrame completo e IFs em memória.

    Não lê ou grava arquivos; o chamador escolhe quando persistir o DataFrame.
    A mesma categoria produz uma linha por (date, horizon_days).
    """
    if test_months <= 0:
        raise ValueError("test_months deve ser positivo")
    if not horizons or any(not isinstance(h, int) or h <= 0 for h in horizons):
        raise ValueError("horizons deve conter inteiros positivos")

    series = DataQuality.sales_series_from_dataframe(df_salesdaily, category,"quantity")
    cutoff = series.index.max() - pd.DateOffset(months=test_months)
    frames: list[pd.DataFrame] = []
    artifacts: dict[int, dict] = {}

    for horizon in sorted(set(horizons)):
        features = make_features(series, category, horizon)
        result, artifact = fit_if_train_only(features, cutoff, contamination)
        ordered = (
            ["date", "category", "horizon_days", "forecast_origin", "split", TARGET_COL]
            + feature_columns(result)
        )
        frames.append(result.loc[:, ordered])
        artifacts[horizon] = artifact

    df_features = pd.concat(frames, ignore_index=True)
    df_features = df_features.sort_values(["horizon_days", "date"]).reset_index(drop=True)
    return df_features, artifacts


def run(
    csv_path: Path = ROOT / "data/raw/salesdaily.csv",
    category: str = "N02BE",
    horizons: tuple[int, ...] | list[int] = (7, 15),
    test_months: int = 6,
    contamination: float = 0.03,
    output_path: Path | None = None,
    save_models: bool = False,
    model_dir: Path = ROOT / "models/anomalydetection",
) -> pd.DataFrame:
    """Lê raw CSV 1 vez -> transforma em RAM -> salva UM Parquet."""
    df_salesdaily = pd.read_csv(csv_path)
    df_features, artifacts = build_features_dataframe(
        df_salesdaily, category=category, horizons=horizons,
        test_months=test_months, contamination=contamination,
    )
    output_path = Path(output_path) if output_path is not None else (
        ROOT / "data/processed" / f"salesdaily_features_{category}.parquet"
    )
    if output_path.suffix.lower() != ".parquet":
        raise ValueError("output_path precisa terminar em .parquet")
    # Usa a infraestrutura já existente no projeto.
    repo = ParquetRepository(base_path=output_path.parent)
    repo.save(df_features, output_path.name, compression="snappy", index=False)

    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)
        for horizon, artifact in artifacts.items():
            path = model_dir / f"{category}_h{horizon:02d}_if.joblib"
            joblib.dump(artifact, path, compress=3)
            print(f"[OK] IF salvo: {path}")

    print(f"[OK] DataFrame: {df_features.shape[0]} linhas, {df_features.shape[1]} colunas")
    print(df_features.groupby(["horizon_days", "split"], observed=True).size())
    return df_features


def main() -> None:
    parser = argparse.ArgumentParser(description="PharmaDemand | DataFrame + IF -> Parquet")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw/salesdaily.csv")
    parser.add_argument("--category", choices=ATC_CATEGORIES, default="N02BE")
    parser.add_argument("--horizons", nargs="+", type=int, default=[7, 15])
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-models", action="store_true",
                        help="Opcional: persiste os IsolationForests como .joblib")
    args = parser.parse_args()
    run(
        csv_path=args.csv, category=args.category, horizons=args.horizons,
        test_months=args.test_months, contamination=args.contamination,
        output_path=args.output, save_models=args.save_models,
    )


if __name__ == "__main__":
    main()
