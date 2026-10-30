"""PharmaDemand: previsao univariada AutoMFLES (Nixtla/StatsForecast).

Interface semelhante a LightGBMRegressorAnomaly:
  gridSearchFind, fit_with_best_params, predict, evaluate,
  save_model, load_model, save_best_params, load_best_params, save_metrics.

O AutoMFLES recebe uma serie temporal DIARIA de vendas reais, sem features
supervisionadas e sem linhas duplicadas por horizon_days. Na saida, a data
representa sempre a data-alvo. O treino e o backtest respeitam as origens:
nenhuma observacao posterior a forecast_origin e usada para ajustar o modelo.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

try:
    from statsforecast.models import AutoMFLES
except ImportError as exc:
    raise ImportError("Instale a dependencia com: pip install statsforecast") from exc


def _prepare(df: pd.DataFrame, date_col: str, target_col: Optional[str],
             series_col: Optional[str]) -> pd.DataFrame:
    columns = [date_col] + ([series_col] if series_col else []) + ([target_col] if target_col else [])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    out = df[columns].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    if out[date_col].isna().any() or (out[date_col] != out[date_col].dt.normalize()).any():
        raise ValueError(f"{date_col}: datas nulas ou com horario; use datas diarias normalizadas.")
    if out[date_col].dt.tz is not None:
        raise ValueError("Datas precisam estar sem timezone.")
    if series_col:
        if out[series_col].isna().any():
            raise ValueError("Categoria de serie nula.")
        out[series_col] = out[series_col].astype(str)
    if target_col:
        out[target_col] = pd.to_numeric(out[target_col], errors="raise")
        if not np.isfinite(out[target_col].to_numpy(dtype=float)).all():
            raise ValueError("Venda real contem NaN ou valores infinitos. Nao interpole o alvo.")
    sort_cols = ([series_col] if series_col else []) + [date_col]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _check_regular(df: pd.DataFrame, date_col: str, series_col: Optional[str]) -> None:
    groups = df.groupby(series_col, sort=False) if series_col else [("serie", df)]
    for name, part in groups:
        dates = pd.DatetimeIndex(part[date_col]).sort_values()
        if dates.has_duplicates:
            raise ValueError(f"Datas duplicadas na serie {name}. Filtre horizontes antes de treinar.")
        if len(dates) > 1:
            expected = pd.date_range(dates.min(), dates.max(), freq="D")
            absent = expected.difference(dates)
            if len(absent):
                raise ValueError(f"Serie {name}: {len(absent)} dia(s) faltando, ex.: "
                                 f"{[str(x.date()) for x in absent[:3]]}. "
                                 "Regularize pela fonte original, nao preencha com zeros.")


def _date_cv(df: pd.DataFrame, date_col: str, n_splits: int,
             horizon: int, min_train_days: int):
    days = pd.DatetimeIndex(df[date_col].unique()).sort_values()
    required = min_train_days + n_splits * horizon
    if len(days) < required:
        raise ValueError(f"CV requer {required} datas; foram encontradas {len(days)}.")
    splits = []
    for i in range(n_splits):
        start = len(days) - (n_splits - i) * horizon
        cutoff = days[start]
        train_ix = np.flatnonzero((df[date_col] < cutoff).to_numpy())
        valid_days = days[start:start + horizon]
        valid_ix = np.flatnonzero(df[date_col].isin(valid_days).to_numpy())
        splits.append((train_ix, valid_ix))
    return splits


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Metricas globais. WAPE em percentual; bias = media(pred - real)."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(y) == 0 or y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("y_true e y_pred precisam ter a mesma forma, sem NaN/inf e nao vazios.")
    den = np.abs(y).sum()
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "wape_pct": float(100.0 * np.abs(y - pred).sum() / den) if den else float("nan"),
        "bias": float(np.mean(pred - y)),
        "n_obs": int(len(y)),
    }


def _json_ready(data):
    if isinstance(data, dict):
        return {str(k): _json_ready(v) for k, v in data.items()}
    if isinstance(data, (tuple, list)):
        return [_json_ready(x) for x in data]
    if isinstance(data, np.generic):
        return _json_ready(data.item())
    if isinstance(data, float) and not np.isfinite(data):
        return None
    if isinstance(data, pd.Timestamp):
        return data.isoformat()
    return data


class _MFLESEstimator(RegressorMixin, BaseEstimator):
    """Adaptador sklearn para GridSearchCV temporal por data e por categoria."""

    def __init__(self, date_col="date", series_col=None, horizon=15,
                 test_size=15, season_length=(7,), n_windows=2, metric="mse",
                 inner_step_size=None, clip_nonnegative=True):
        self.date_col = date_col
        self.series_col = series_col
        self.horizon = horizon
        self.test_size = test_size
        self.season_length = season_length
        self.n_windows = n_windows
        self.metric = metric
        self.inner_step_size = inner_step_size
        self.clip_nonnegative = clip_nonnegative

    def _new_model(self):
        season = list(self.season_length) if isinstance(self.season_length, (tuple, list)) else self.season_length
        return AutoMFLES(
            test_size=self.test_size,
            season_length=season,
            n_windows=self.n_windows,
            metric=self.metric,
            step_size=self.inner_step_size,
            verbose=False,
        )

    def fit(self, X, y):
        if self.horizon < 1 or self.test_size < 1:
            raise ValueError("horizon/test_size devem ser positivos.")
        if len(X) != len(y):
            raise ValueError("X e y nao tem o mesmo numero de observacoes.")
        data = X.copy()
        data["__target__"] = np.asarray(y, dtype=float)
        data = _prepare(data, self.date_col, "__target__", self.series_col)
        _check_regular(data, self.date_col, self.series_col)
        self.models_ = {}
        self.last_dates_ = {}
        groups = data.groupby(self.series_col, sort=False) if self.series_col else [("__single__", data)]
        for key, part in groups:
            if len(part) < max(30, self.test_size * (self.n_windows + 1) + 1):
                raise ValueError(f"Historico insuficiente para AutoMFLES na serie {key}.")
            model = self._new_model()
            model.fit(part["__target__"].to_numpy(dtype=float))
            self.models_[str(key)] = model
            self.last_dates_[str(key)] = pd.Timestamp(part[self.date_col].iloc[-1])
        return self

    def predict(self, X):
        if not hasattr(self, "models_"):
            raise ValueError("Treine o modelo antes de prever.")
        cols = [self.date_col] + ([self.series_col] if self.series_col else [])
        if any(col not in X for col in cols):
            raise ValueError(f"Previsao requer colunas {cols}.")
        work = _prepare(X, self.date_col, None, self.series_col)
        # _prepare ordena: preservar a ordem original da entrada via chave de linha.
        raw = X[cols].copy().reset_index(drop=True)
        raw[self.date_col] = pd.to_datetime(raw[self.date_col])
        if self.series_col:
            raw[self.series_col] = raw[self.series_col].astype(str)
        raw["__row__"] = np.arange(len(raw))
        predicted = np.empty(len(raw), dtype=float)
        groups = raw.groupby(self.series_col, sort=False) if self.series_col else [("__single__", raw)]
        for name, part in groups:
            key = str(name)
            if key not in self.models_:
                raise ValueError(f"Categoria nao treinada: {key}")
            offset = (part[self.date_col] - self.last_dates_[key]).dt.days.to_numpy()
            if (offset < 1).any() or (offset > self.horizon).any():
                raise ValueError(f"Previsoes para {key} devem estar entre D+1 e D+{self.horizon} "
                                 f"apos {self.last_dates_[key].date()}.")
            h = int(offset.max())
            means = np.asarray(self.models_[key].predict(h=h)["mean"], dtype=float)
            values = means[offset - 1]
            if self.clip_nonnegative:
                values = np.maximum(values, 0.0)
            predicted[part["__row__"].to_numpy(dtype=int)] = values
        if not np.isfinite(predicted).all():
            raise ValueError("AutoMFLES produziu previsoes nao finitas.")
        return predicted


class AutoMFLESForecaster:
    """Wrapper PharmaDemand com GridSearch temporal e backtest rigoroso D+7/D+15."""

    def __init__(self, target_col: str, numeric_features: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None,
                 model_dir: str = "models/baseline", random_state: int = 42,
                 n_splits: int = 3, scoring: str = "neg_root_mean_squared_error",
                 date_col: str = "date", series_col: Optional[str] = None,
                 forecast_horizon: int = 15, min_train_days: int = 90,
                 clip_nonnegative: bool = True):
        self.target_col = target_col
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        self.n_splits = n_splits
        self.scoring = scoring
        self.date_col = date_col
        self.series_col = series_col
        self.forecast_horizon = forecast_horizon
        self.min_train_days = min_train_days
        self.clip_nonnegative = clip_nonnegative
        self.grid_search = None
        self.best_model = None
        self.best_params = None
        self.pipeline = None
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.numeric_features or self.categorical_features:
            warnings.warn("AutoMFLES e univariado nesta implementacao: features de LightGBM/IF "
                          "nao sao usadas.", UserWarning, stacklevel=2)

    def _build_pipeline(self):
        return Pipeline([("model", _MFLESEstimator(
            date_col=self.date_col, series_col=self.series_col,
            horizon=self.forecast_horizon, test_size=self.forecast_horizon,
            clip_nonnegative=self.clip_nonnegative,
        ))])

    @staticmethod
    def _default_param_grid():
        # AutoMFLES faz CV/otimizacao INTERNA: grade externa moderada.
        return {
            "model__season_length": [(7,), (7, 30)],
            "model__n_windows": [2],
            "model__metric": ["mse"],
        }

    def gridSearchFind(self, df_train: pd.DataFrame,
                       param_grid: Optional[Dict[str, List[Any]]] = None) -> GridSearchCV:
        data = _prepare(df_train, self.date_col, self.target_col, self.series_col)
        _check_regular(data, self.date_col, self.series_col)
        columns = [self.date_col] + ([self.series_col] if self.series_col else [])
        cv = _date_cv(data, self.date_col, self.n_splits,
                      self.forecast_horizon, self.min_train_days)
        self.pipeline = self._build_pipeline()
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid if param_grid is not None else self._default_param_grid(),
            scoring=self.scoring, cv=cv, n_jobs=1, verbose=1,
            refit=True, error_score="raise", return_train_score=False,
        )
        self.grid_search.fit(data[columns], data[self.target_col].to_numpy())
        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        return self.grid_search

    def fit_with_best_params(self, df_train: pd.DataFrame,
                             params_path: Optional[str] = None):
        if params_path:
            self.load_best_params(params_path)
        if self.best_params is None:
            raise ValueError("Primeiro execute gridSearchFind() ou load_best_params().")
        data = _prepare(df_train, self.date_col, self.target_col, self.series_col)
        _check_regular(data, self.date_col, self.series_col)
        columns = [self.date_col] + ([self.series_col] if self.series_col else [])
        self.pipeline = self._build_pipeline().set_params(**self.best_params)
        self.pipeline.fit(data[columns], data[self.target_col].to_numpy())
        self.best_model = self.pipeline
        return self.best_model

    def predict(self, df_future: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Modelo nao treinado.")
        columns = [self.date_col] + ([self.series_col] if self.series_col else [])
        return self.best_model.predict(df_future[columns])

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        """Somente os proximos <=forecast_horizon dias apos o fit atual."""
        data = _prepare(df_test, self.date_col, self.target_col, self.series_col)
        return calculate_metrics(data[self.target_col], self.predict(data))

    def _validated_union(self, df_train, df_test) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = _prepare(df_train, self.date_col, self.target_col, self.series_col)
        test = _prepare(df_test, self.date_col, self.target_col, self.series_col)
        _check_regular(train, self.date_col, self.series_col)
        _check_regular(test, self.date_col, self.series_col)
        if test[self.date_col].min() <= train[self.date_col].max():
            raise ValueError("Teste deve comecar depois do fim do treino (sem sobreposicao).")
        combined = pd.concat([train, test], ignore_index=True)
        _check_regular(combined, self.date_col, self.series_col)
        return train, test, combined

    def backtest(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 step_size: Optional[int] = None) -> pd.DataFrame:
        """Backtest em blocos: D+1 .. D+h; NAO e comparacao exclusivamente D+15."""
        if self.best_params is None:
            raise ValueError("Execute gridSearchFind ou load_best_params no treino.")
        step = self.forecast_horizon if step_size is None else int(step_size)
        if not 1 <= step <= self.forecast_horizon:
            raise ValueError("step_size fora do intervalo 1..forecast_horizon")
        _, test, combined = self._validated_union(df_train, df_test)
        dates = pd.DatetimeIndex(test[self.date_col].unique()).sort_values()
        cols = [self.date_col] + ([self.series_col] if self.series_col else [])
        parts = []
        for begin in range(0, len(dates), step):
            days = dates[begin:begin + step]
            origin = days[0] - pd.Timedelta(days=1)
            history = combined.loc[combined[self.date_col] <= origin]
            block = test.loc[test[self.date_col].isin(days)].copy()
            est = self._build_pipeline().set_params(**self.best_params)
            est.fit(history[cols], history[self.target_col].to_numpy())
            block["y_pred"] = est.predict(block[cols])
            block["forecast_origin"] = origin
            block["horizon_days"] = (block[self.date_col] - origin).dt.days
            parts.append(block)
        return pd.concat(parts, ignore_index=True)

    def evaluate_backtest(self, df_train, df_test, step_size=None) -> Dict[str, float]:
        result = self.backtest(df_train, df_test, step_size=step_size)
        return calculate_metrics(result[self.target_col], result["y_pred"])

    def backtest_fixed_horizons(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                                horizons: Sequence[int] = (7, 15),
                                step_size: int = 7, verbose: bool = False) -> pd.DataFrame:
        """Predicoes com alvos iguais em D+7 e D+15, sem leakage.

        Para cada target date T, os horizontes h usam origin=T-h. Reajusta a
        serie com vendas observadas ate origin. step_size=1 cobre TODOS os dias
        do holdout (mais custoso); step_size=7 e amostra semanal exploratoria.
        Usa um unico ajuste por origin, mesmo quando h=7 e h=15 compartilham
        a origem. Jamais busca hiperparametros no periodo de teste.
        """
        if self.best_params is None:
            raise ValueError("Execute gridSearchFind/load_best_params somente no treino.")
        horizons = tuple(sorted(set(int(h) for h in horizons)))
        if not horizons or min(horizons) < 1 or max(horizons) > self.forecast_horizon:
            raise ValueError(f"horizons devem estar em 1..{self.forecast_horizon}.")
        if step_size < 1:
            raise ValueError("step_size precisa ser >= 1.")
        _, test, combined = self._validated_union(df_train, df_test)
        cols = [self.date_col] + ([self.series_col] if self.series_col else [])
        all_dates = pd.DatetimeIndex(test[self.date_col].unique()).sort_values()
        sampled_dates = all_dates[::step_size]
        requests = {}
        for day in sampled_dates:
            for h in horizons:
                origin = day - pd.Timedelta(days=h)
                requests.setdefault(origin, []).append((day, h))
        output = []
        for j, (origin, targets) in enumerate(sorted(requests.items()), start=1):
            hist = combined.loc[combined[self.date_col] <= origin].copy()
            if hist[self.date_col].nunique() < self.min_train_days:
                raise ValueError(f"Historico insuficiente na origem {origin.date()}.")
            estimator = self._build_pipeline().set_params(**self.best_params)
            estimator.fit(hist[cols], hist[self.target_col].to_numpy())
            if verbose and (j == 1 or j % 10 == 0 or j == len(requests)):
                print(f"Ajustes do backtest: {j}/{len(requests)} (origem {origin.date()})")
            for day, h in targets:
                actuals = test.loc[test[self.date_col] == day]
                predict_frame = actuals[cols]
                yhats = estimator.predict(predict_frame)
                for (_, actual), pred in zip(actuals.iterrows(), yhats):
                    row = {self.date_col: day, self.target_col: float(actual[self.target_col]),
                           "forecast_origin": origin, "horizon_days": h, "y_pred": float(pred),
                           "model": "AutoMFLES"}
                    if self.series_col:
                        row[self.series_col] = str(actual[self.series_col])
                    output.append(row)
        result = pd.DataFrame(output)
        order = ([self.series_col] if self.series_col else []) + [self.date_col, "horizon_days"]
        return result.sort_values(order).reset_index(drop=True)

    def metrics_by_horizon(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        if "horizon_days" not in forecasts:
            raise ValueError("Coluna horizon_days ausente.")
        metrics = []
        for h, part in forecasts.groupby("horizon_days", sort=True):
            metrics.append({"horizon_days": int(h), **calculate_metrics(
                part[self.target_col], part["y_pred"])})
        return pd.DataFrame(metrics)

    def save_best_params(self, filename: str = "automfles_best_params.json") -> Path:
        if self.best_params is None:
            raise ValueError("Melhores parametros ainda nao definidos.")
        path = self.model_dir / filename
        data = {"best_params": self.best_params,
                "best_score": float(self.grid_search.best_score_) if self.grid_search else None,
                "scoring": self.scoring, "n_splits": self.n_splits,
                "forecast_horizon": self.forecast_horizon,
                "date_col": self.date_col, "series_col": self.series_col,
                "target_col": self.target_col,
                "cv_strategy": "expanding_window_time_series"}
        path.write_text(json.dumps(_json_ready(data), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_best_params(self, params_path: str) -> Dict[str, Any]:
        data = json.loads(Path(params_path).read_text(encoding="utf-8"))
        if data.get("forecast_horizon") != self.forecast_horizon or data.get("series_col") != self.series_col:
            raise ValueError("Parametros salvos nao correspondem a horizonte/serie da instancia.")
        self.best_params = data["best_params"]
        # json serializa tupla como lista; sklearn/AutoMFLES aceitam lista.
        return self.best_params

    def save_metrics(self, metrics: Dict[str, float],
                     filename: str = "automfles_metrics.json") -> Path:
        path = self.model_dir / filename
        path.write_text(json.dumps(_json_ready(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_model(self, filename: str = "automfles_model.joblib") -> Path:
        if self.best_model is None:
            raise ValueError("Modelo ainda nao treinado.")
        path = self.model_dir / filename
        joblib.dump({"pipeline": self.best_model,
                     "best_params": self.best_params,
                     "date_col": self.date_col, "series_col": self.series_col,
                     "target_col": self.target_col,
                     "forecast_horizon": self.forecast_horizon,
                     "clip_nonnegative": self.clip_nonnegative}, path)
        return path

    def load_model(self, model_path: str) -> None:
        data = joblib.load(model_path)  # Apenas modelos produzidos por fontes confiaveis.
        self.best_model = data["pipeline"]
        self.pipeline = self.best_model
        self.best_params = data["best_params"]
        self.date_col = data["date_col"]
        self.series_col = data["series_col"]
        self.target_col = data["target_col"]
        self.forecast_horizon = data["forecast_horizon"]
        self.clip_nonnegative = data["clip_nonnegative"]

    def get_model(self):
        if self.best_model is None:
            raise ValueError("Modelo ainda nao treinado.")
        inner = self.best_model.named_steps["model"].models_
        return next(iter(inner.values())) if len(inner) == 1 else inner

    def get_model_type(self):
        return type(self.get_model())
