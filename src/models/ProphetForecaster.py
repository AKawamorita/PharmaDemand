"""Forecast diário por categoria ATC com GridSearchCV temporal.

Adaptação da interface de LightGBMRegressorAnomaly.py.
Importante: este modelo é UNIVARIADO. Features de IF/LightGBM não são
consumidas, pois não se conhecem necessariamente seus valores futuros.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prophet import Prophet


def _json_compatible(value):
    """Converte np.* e timestamps para tipos que json.dump aceita."""
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_compatible(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _prepare(df: pd.DataFrame, date_col: str, target_col: str | None,
             series_col: str | None, require_target=True) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Forneça um pandas.DataFrame.")
    expected = [date_col] + ([series_col] if series_col else [])
    if require_target:
        expected.append(target_col)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    if out[date_col].isna().any():
        raise ValueError(f"Datas ausentes em {date_col}")
    if (out[date_col] != out[date_col].dt.normalize()).any():
        raise ValueError("Série diária exige datas sem horário (00:00).")
    if out[date_col].dt.tz is not None:
        raise ValueError("Datas precisam estar sem fuso (naive), frequência diária.")
    if series_col:
        if out[series_col].isna().any():
            raise ValueError("Identificadores de categorias ATC não podem ser nulos.")
        out[series_col] = out[series_col].astype(str)
    if require_target:
        out[target_col] = pd.to_numeric(out[target_col], errors="raise")
        if not np.isfinite(out[target_col].to_numpy(dtype=float)).all():
            raise ValueError("Target possui NaN ou valores infinitos.")
    return out.sort_values([date_col] + ([series_col] if series_col else []),
                           kind="mergesort").reset_index(drop=True)


def _date_splits(df: pd.DataFrame, date_col: str, n_splits: int,
                 horizon: int, min_train_days: int):
    """Expanding-window; cada fold prevê o próximo bloco de `horizon` dias."""
    dates = pd.DatetimeIndex(df[date_col].unique()).sort_values()
    if len(dates) < min_train_days + n_splits * horizon:
        raise ValueError(
            f"São necessários ao menos {min_train_days + n_splits * horizon} "
            f"dias únicos para {n_splits} folds de {horizon} dias; há {len(dates)}."
        )
    indices = []
    for fold in range(n_splits):
        begin = len(dates) - (n_splits - fold) * horizon
        end = begin + horizon
        train_mask = df[date_col] < dates[begin]
        test_mask = df[date_col].between(dates[begin], dates[end - 1])
        train_ix = np.flatnonzero(train_mask.to_numpy())
        test_ix = np.flatnonzero(test_mask.to_numpy())
        if not len(train_ix) or not len(test_ix):
            raise ValueError("Fold temporal vazio.")
        indices.append((train_ix, test_ix))
    return indices


def _validate_regular_series(frame: pd.DataFrame, date_col: str, series_col: str | None):
    keys = [series_col] if series_col else None
    groups = frame.groupby(keys, sort=False) if keys else [("__single__", frame)]
    for ident, g in groups:
        ordered = pd.DatetimeIndex(g[date_col]).sort_values()
        if ordered.has_duplicates:
            raise ValueError(f"Datas duplicadas na série {ident}.")
        if len(ordered) > 1:
            expected = pd.date_range(ordered[0], ordered[-1], freq="D")
            if len(expected) != len(ordered):
                absent = expected.difference(ordered)
                raise ValueError(
                    f"Série {ident} com {len(absent)} dia(s) faltando, "
                    f"exemplo: {list(absent[:3].date)}. Regularize fora da classe "
                    "conforme regra de negócio; não complete automaticamente com zero."
                )


def _metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    denom = float(np.abs(y).sum())
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else float("nan"),
        "wape_pct": float(100 * np.abs(err).sum() / denom) if denom else float("nan"),
        "bias": float(np.mean(err)),  # positivo: superestimação
    }


class _PanelEstimator(RegressorMixin, BaseEstimator):
    """Estimador sklearn interno para GridSearchCV respeitando as categorias."""
    def __init__(self, date_col="date", series_col=None, horizon=15,
                 clip_nonnegative=True, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                 seasonality_mode="additive", weekly_seasonality=True,
                 yearly_seasonality="auto"):
        self.date_col = date_col
        self.series_col = series_col
        self.horizon = horizon
        self.clip_nonnegative = clip_nonnegative
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality

    def _new_model(self):
        return Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=False, uncertainty_samples=0
        )

    def fit(self, X, y):
        # Associa y a X ANTES de ordenar para preservar o alinhamento.
        y = np.asarray(y, dtype=float)
        if len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError("Dimensão do target inválida ou valores não finitos.")
        if self.horizon < 1:
            raise ValueError("horizon deve ser >= 1")
        frame = X.copy()
        frame["__target_forecast__"] = y
        frame = _prepare(frame, self.date_col, "__target_forecast__",
                         self.series_col, require_target=True)
        _validate_regular_series(frame, self.date_col, self.series_col)
        self.models_ = {}
        self.last_dates_ = {}
        groups = (frame.groupby(self.series_col, sort=False) if self.series_col
                  else [("__single__", frame)])
        for ident, sub in groups:
            sub = sub.sort_values(self.date_col)
            key = str(ident)
            model = self._new_model()
            model.fit(sub[[self.date_col, "__target_forecast__"]].rename(
                columns={self.date_col: "ds", "__target_forecast__": "y"}
            ))
            self.models_[key] = model
            self.last_dates_[key] = pd.Timestamp(sub[self.date_col].iloc[-1])
        return self

    def predict(self, X):
        if not hasattr(self, "models_"):
            raise ValueError("Treine o modelo antes de prever.")
        expected = [self.date_col] + ([self.series_col] if self.series_col else [])
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(f"Colunas ausentes: {missing}")
        df = X[expected].copy().reset_index(drop=True)
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="raise")
        if df[self.date_col].isna().any():
            raise ValueError("Datas futuras nulas.")
        if self.series_col:
            if df[self.series_col].isna().any():
                raise ValueError("Categoria ATC nula.")
            df[self.series_col] = df[self.series_col].astype(str)
        df["__idx"] = np.arange(len(df))
        output = np.empty(len(df), dtype=float)
        groups = (df.groupby(self.series_col, sort=False) if self.series_col
                  else [("__single__", df)])
        for ident, sub in groups:
            key = str(ident)
            if key not in self.models_:
                raise ValueError(f"Série {key} não encontrada no treinamento.")
            last = self.last_dates_[key]
            offsets = (sub[self.date_col] - last).dt.days.to_numpy()
            if (offsets < 1).any() or (offsets > self.horizon).any():
                raise ValueError(
                    f"Datas da série {key} precisam ser posteriores a {last.date()} "
                    f"e estar até D+{self.horizon}. "
                    "Para avaliar meses de teste, use backtest(df_train, df_test)."
                )
            values = self.models_[key].predict(
                pd.DataFrame({"ds": sub[self.date_col].to_numpy()})
            )["yhat"].to_numpy(dtype=float)
            if self.clip_nonnegative:
                values = np.maximum(values, 0.0)
            output[sub["__idx"].to_numpy(dtype=int)] = values
        return output


class ProphetForecaster:
    """API semelhante à LightGBMRegressorAnomaly, com validação D+7/D+15.

    - gridSearchFind(df_train) -> GridSearchCV temporal (sem embaralhamento)
    - fit_with_best_params(df_train) -> Pipeline ajustado
    - predict(df_future) -> ndarray, com D+1..D+h por categoria
    - evaluate(df_test) -> métricas D+1..D+h
    - backtest(df_train, df_test) -> DataFrame para holdout longo em blocos
    """
    def __init__(self, target_col: str, numeric_features: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None,
                 model_dir: str = "models/baseline", random_state: int = 42,
                 n_splits: int = 3,
                 scoring: str = "neg_root_mean_squared_error",
                 date_col: str = "date", series_col: Optional[str] = None,
                 forecast_horizon: int = 15, min_train_days: int = 90,
                 clip_nonnegative: bool = True):
        self.target_col = target_col
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.model_dir = Path(model_dir)
        self.random_state = random_state  # Interface compatível (não usado pelo método)
        self.n_splits = n_splits
        self.scoring = scoring
        self.date_col = date_col
        self.series_col = series_col
        self.forecast_horizon = forecast_horizon
        self.min_train_days = min_train_days
        self.clip_nonnegative = clip_nonnegative
        self.pipeline = None
        self.grid_search = None
        self.best_model = None
        self.best_params = None
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.numeric_features or self.categorical_features:
            warnings.warn(
                "ProphetForecaster é univariado: numeric_features/categorical_features "
                "são aceitas por compatibilidade, mas NÃO são utilizadas. "
                "Anomaly_score/IF não entram neste modelo.", UserWarning, stacklevel=2)

    def _build_pipeline(self):
        return Pipeline([("model", _PanelEstimator(
            date_col=self.date_col, series_col=self.series_col,
            horizon=self.forecast_horizon,
            clip_nonnegative=self.clip_nonnegative,
        ))])

    def _default_param_grid(self) -> Dict[str, List[Any]]:
        return {"model__changepoint_prior_scale": [0.01, 0.05, 0.1],
                "model__seasonality_prior_scale": [0.1, 1.0],
                "model__seasonality_mode": ["additive", "multiplicative"]}

    def gridSearchFind(self, df_train: pd.DataFrame,
                       param_grid: Optional[Dict[str, List[Any]]] = None) -> GridSearchCV:
        frame = _prepare(df_train, self.date_col, self.target_col, self.series_col)
        _validate_regular_series(frame, self.date_col, self.series_col)
        cv = _date_splits(frame, self.date_col, self.n_splits,
                          self.forecast_horizon, self.min_train_days)
        columns = [self.date_col] + ([self.series_col] if self.series_col else [])
        self.pipeline = self._build_pipeline()
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid if param_grid is not None else self._default_param_grid(),
            scoring=self.scoring, cv=cv, n_jobs=1, verbose=1,
            refit=True, error_score="raise", return_train_score=False,
        )
        self.grid_search.fit(frame[columns], frame[self.target_col].to_numpy())
        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        return self.grid_search

    def fit_with_best_params(self, df_train: pd.DataFrame,
                             params_path: Optional[str] = None):
        if params_path is not None:
            self.load_best_params(params_path)
        if self.best_params is None:
            raise ValueError("Execute gridSearchFind() ou load_best_params() antes.")
        frame = _prepare(df_train, self.date_col, self.target_col, self.series_col)
        columns = [self.date_col] + ([self.series_col] if self.series_col else [])
        self.pipeline = self._build_pipeline().set_params(**self.best_params)
        self.pipeline.fit(frame[columns], frame[self.target_col].to_numpy())
        self.best_model = self.pipeline
        return self.best_model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Execute gridSearchFind() ou fit_with_best_params() antes.")
        cols = [self.date_col] + ([self.series_col] if self.series_col else [])
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes: {missing}")
        return self.best_model.predict(df[cols])

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        """Avalia apenas um horizonte futuro <= forecast_horizon."""
        for col in [self.target_col]:
            if col not in df_test.columns:
                raise ValueError(f"Coluna ausente: {col}")
        frame = df_test.reset_index(drop=True)
        y_hat = self.predict(frame)
        return _metrics(frame[self.target_col].to_numpy(dtype=float), y_hat)

    def backtest(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 step_size: Optional[int] = None) -> pd.DataFrame:
        """Rolling-origin: treina somente até cada origem e prevê próximo bloco.

        A cada novo bloco incorpora valores REAIS anteriores (retraining).
        A comparação final usa o holdout sem busca de hiperparâmetros nele.
        Retorna df_test + y_pred, forecast_origin, horizon_step (linha a linha).
        """
        if self.best_params is None:
            raise ValueError("Primeiro encontre/carregue best_params no TREINO.")
        stride = step_size or self.forecast_horizon
        if not (1 <= stride <= self.forecast_horizon):
            raise ValueError("step_size deve estar entre 1 e forecast_horizon.")
        train = _prepare(df_train, self.date_col, self.target_col, self.series_col)
        test = _prepare(df_test, self.date_col, self.target_col, self.series_col)
        if test[self.date_col].min() <= train[self.date_col].max():
            raise ValueError("O teste deve começar após o fim do treino.")
        combined = pd.concat([train, test], ignore_index=True)
        _validate_regular_series(combined, self.date_col, self.series_col)
        # Exige um painel com cobertura completa por categoria em todas as origens.
        unique_days = pd.DatetimeIndex(test[self.date_col].unique()).sort_values()
        cols = [self.date_col] + ([self.series_col] if self.series_col else [])
        outputs = []
        for start in range(0, len(unique_days), stride):
            block_days = unique_days[start:start + stride]
            origin = block_days[0] - pd.Timedelta(days=1)
            historical = combined.loc[combined[self.date_col] <= origin].copy()
            block = test.loc[test[self.date_col].isin(block_days)].copy()
            estimator = self._build_pipeline().set_params(**self.best_params)
            estimator.fit(historical[cols], historical[self.target_col].to_numpy())
            block["y_pred"] = estimator.predict(block[cols])
            block["forecast_origin"] = origin
            block["horizon_step"] = (block[self.date_col] - origin).dt.days
            outputs.append(block)
        return pd.concat(outputs, ignore_index=True)

    def evaluate_backtest(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                          step_size: Optional[int] = None) -> Dict[str, float]:
        result = self.backtest(df_train, df_test, step_size=step_size)
        return _metrics(result[self.target_col], result["y_pred"])

    def save_best_params(self, filename: str = "prophet_best_params.json") -> Path:
        if self.best_params is None:
            raise ValueError("best_params indisponíveis.")
        out = self.model_dir / filename
        payload = {
            "best_params": self.best_params,
            "best_score": (float(self.grid_search.best_score_) if self.grid_search is not None else None),
            "scoring": self.scoring, "n_splits": self.n_splits,
            "forecast_horizon": self.forecast_horizon,
            "target_col": self.target_col, "date_col": self.date_col,
            "series_col": self.series_col,
            "cv_strategy": "expanding_window_fixed_future_horizon",
        }
        out.write_text(json.dumps(_json_compatible(payload), indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        return out

    def load_best_params(self, params_path: str) -> Dict[str, Any]:
        data = json.loads(Path(params_path).read_text(encoding="utf-8"))
        if data.get("forecast_horizon", self.forecast_horizon) != self.forecast_horizon:
            raise ValueError("Horizonte do arquivo diferente do modelo em uso.")
        if data.get("series_col", self.series_col) != self.series_col:
            raise ValueError("series_col do arquivo não corresponde à instância.")
        self.best_params = data["best_params"]
        return self.best_params

    def save_metrics(self, metrics: Dict[str, float],
                     filename: str = "prophet_metrics.json") -> Path:
        out = self.model_dir / filename
        out.write_text(json.dumps(_json_compatible(metrics), indent=2,
                                  allow_nan=False), encoding="utf-8")
        return out

    def get_model(self):
        if self.best_model is None:
            raise ValueError("Modelo indisponível.")
        estimators = self.best_model.named_steps["model"].models_
        return next(iter(estimators.values())) if len(estimators) == 1 else estimators

    def get_model_type(self):
        return type(self.get_model())

    def save_model(self, filename: str = "prophet_model.joblib") -> Path:
        """Usa serialização JSON nativa do Prophet dentro de bundle joblib."""
        if self.best_model is None:
            raise ValueError("Modelo não treinado.")
        from prophet.serialize import model_to_json
        est = self.best_model.named_steps["model"]
        payload = {
            "models_json": {k: model_to_json(m) for k, m in est.models_.items()},
            "last_dates": {k: dt.isoformat() for k, dt in est.last_dates_.items()},
            "best_params": self.best_params,
            "date_col": self.date_col, "series_col": self.series_col,
            "forecast_horizon": self.forecast_horizon,
            "target_col": self.target_col,
            "clip_nonnegative": self.clip_nonnegative,
        }
        path = self.model_dir / filename
        joblib.dump(payload, path)
        return path

    def load_model(self, model_path: str) -> None:
        from prophet.serialize import model_from_json
        data = joblib.load(model_path)  # Carregue apenas artefatos confiáveis.
        self.date_col = data["date_col"]
        self.series_col = data["series_col"]
        self.target_col = data["target_col"]
        self.forecast_horizon = data["forecast_horizon"]
        self.clip_nonnegative = data["clip_nonnegative"]
        self.best_params = data["best_params"]
        pipeline = self._build_pipeline()
        if self.best_params is not None:
            pipeline.set_params(**self.best_params)
        estimator = pipeline.named_steps["model"]
        estimator.models_ = {k: model_from_json(s) for k, s in data["models_json"].items()}
        estimator.last_dates_ = {k: pd.Timestamp(d) for k, d in data["last_dates"].items()}
        self.pipeline = pipeline
        self.best_model = pipeline

