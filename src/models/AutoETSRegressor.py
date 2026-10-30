"""AutoETS (Nixtla StatsForecast) para UMA série diária do PharmaDemand.

Instalação: pip install statsforecast
AutoETS seleciona automaticamente componentes ETS por critério de informação;
gridSearchFind compara poucas configurações externas via backtesting temporal.
AutoETS NÃO consome colunas exógenas. Use numeric_features=[] neste modelo.

Calendário: por padrão rejeita dias faltantes. Use missing_dates="zero" SOMENTE
quando uma data não registrada representar comprovadamente zero vendas.
A classe nunca interpola os alvos usando observações futuras.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsforecast.models import AutoETS
except ImportError:
    AutoETS = None


class AutoETSRegressor:
    """Forecast univariado ETS com validação expanding-window por horizonte."""

    def __init__(
        self,
        target_col: str,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        model_dir: str = "models/baseline",
        random_state: int = 42,
        n_splits: int = 3,
        scoring: str = "neg_root_mean_squared_error",
        date_col: str = "date",
        horizon: int = 15,
        season_length: int = 7,
        freq: str = "D",
        clip_negative: bool = True,
        missing_dates: str = "raise",
    ):
        if numeric_features or categorical_features:
            raise ValueError(
                "AutoETS é univariado: passe numeric_features=[] e categorical_features=None. "
                "Ele não usa features do Isolation Forest, calendário ou XGBoost."
            )
        if n_splits < 2 or horizon < 1 or season_length < 1:
            raise ValueError("Exige n_splits>=2, horizon>=1, season_length>=1.")
        if scoring not in {"neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"}:
            raise ValueError(f"Scoring não suportado: {scoring}")
        self.target_col = target_col
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.model_dir = Path(model_dir)
        self.random_state = random_state  # Mantido por compatibilidade.
        self.n_splits = n_splits
        self.scoring = scoring
        self.date_col = date_col
        self.horizon = horizon
        self.season_length = season_length
        self.freq = freq
        self.clip_negative = clip_negative
        if missing_dates not in {"raise", "zero"}:
            raise ValueError("missing_dates deve ser 'raise' ou 'zero'.")
        self.missing_dates = missing_dates
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_model = None
        self.grid_search = None
        self.last_train_date: Optional[pd.Timestamp] = None
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _dependency():
        if AutoETS is None:
            raise ImportError("Instale Nixtla StatsForecast: pip install statsforecast")

    def _default_param_grid(self) -> Dict[str, List[Any]]:
        # 'ZZZ' já busca automaticamente componentes ETS por AICc.
        # 2 periodicidades * 2 opções de amortecimento = 4 alternativas.
        return {
            "season_length": [self.season_length, 1],
            "model": ["ZZZ"],
            "damped": [None, True],
        }

    def calendar_diagnostics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verifica lacunas no calendário ANTES de decidir como tratá-las.

        Não modifica o DataFrame. Datas duplicadas continuam sendo erro, pois
        uma instância do AutoETS deve representar uma única série ATC diária.
        """
        if self.date_col not in df.columns:
            raise ValueError(f"Coluna de data ausente: {self.date_col!r}.")
        if df.empty:
            raise ValueError("DataFrame vazio.")
        dates = pd.to_datetime(df[self.date_col], errors="raise").dt.normalize()
        if dates.isna().any():
            raise ValueError("Datas nulas.")
        unique = pd.DatetimeIndex(dates).unique().sort_values()
        complete = pd.date_range(unique[0], unique[-1], freq=self.freq)
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

    def _prepare(self, df: pd.DataFrame, *, target: bool) -> pd.DataFrame:
        required = [self.date_col] + ([self.target_col] if target else [])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes: {missing}")
        if df.empty:
            raise ValueError("DataFrame vazio.")

        data = df.copy()
        data[self.date_col] = pd.to_datetime(data[self.date_col], errors="raise").dt.normalize()
        if data[self.date_col].isna().any():
            raise ValueError("Datas nulas.")
        data = data.sort_values(self.date_col).reset_index(drop=True)
        dates = pd.DatetimeIndex(data[self.date_col])
        if dates.has_duplicates:
            raise ValueError("Datas repetidas. Filtre uma categoria ATC ou agregue a série.")

        if target:
            # Somente série com target: reconstruir calendário completo.
            # Assim cada fold da validação corresponde a dias reais e não apenas
            # às linhas existentes no CSV (que podem pular datas).
            values = pd.to_numeric(data[self.target_col], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError("Target contém nulos, infinitos ou não numéricos.")
            data[self.target_col] = values.astype(float)
            complete = pd.date_range(dates[0], dates[-1], freq=self.freq)
            missing_idx = complete.difference(dates)
            if len(missing_idx):
                if self.missing_dates == "raise":
                    examples = ", ".join(str(d.date()) for d in missing_idx[:8])
                    raise ValueError(
                        f"Série irregular ({self.freq}): {len(missing_idx)} dias ausentes "
                        f"entre {dates[0].date()} e {dates[-1].date()}. "
                        f"Exemplos: {examples}. Use calendar_diagnostics(df). "
                        "Se a ausência representar ZERO vendas, "
                        "defina missing_dates='zero'."
                    )
                # Seguro contra leakage apenas se ausência = 0 vendas verdadeiro.
                data = (
                    data.set_index(self.date_col)
                    .reindex(complete)
                    .rename_axis(self.date_col)
                    .reset_index()
                )
                data[self.target_col] = data[self.target_col].fillna(0.0)
        # target=False: preservar exatamente as datas pedidas pelo usuário.
        # predict() calcula desde o dia seguinte ao cutoff e seleciona as
        # previsões correspondentes, sem confundir 1 linha com 1 dia.
        return data

    def _windows(self, n: int):
        initial = n - self.n_splits * self.horizon
        minimum = max(2 * self.season_length, 20)
        if initial < minimum:
            raise ValueError(
                f"Histórico insuficiente: {n} linhas, mínimo {minimum + self.n_splits * self.horizon}."
            )
        for i in range(self.n_splits):
            cutoff = initial + i * self.horizon
            yield np.arange(cutoff), np.arange(cutoff, cutoff + self.horizon)

    @staticmethod
    def _parameter_candidates(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        allowed = {"season_length", "model", "damped", "phi"}
        if not grid or set(grid) - allowed:
            raise ValueError(f"Chaves permitidas: {sorted(allowed)}")
        if any(not values for values in grid.values()):
            raise ValueError("param_grid contém listas vazias.")
        return [dict(zip(grid, values)) for values in product(*grid.values())]

    def _fit_model(self, y: np.ndarray, params: Dict[str, Any]):
        self._dependency()
        kwargs = {
            "season_length": int(params.get("season_length", self.season_length)),
            "model": params.get("model", "ZZZ"),
            "damped": params.get("damped", None),
        }
        if "phi" in params and params["phi"] is not None:
            kwargs["phi"] = float(params["phi"])
        model = AutoETS(**kwargs)
        model.fit(np.asarray(y, dtype=float))
        return model

    def _forecast(self, fitted, steps: int) -> np.ndarray:
        values = np.asarray(fitted.predict(h=steps)["mean"], dtype=float)
        if len(values) != steps or not np.isfinite(values).all():
            raise ValueError("AutoETS retornou previsão inválida.")
        return np.maximum(values, 0) if self.clip_negative else values

    def _score(self, actual: np.ndarray, pred: np.ndarray) -> float:
        if self.scoring == "neg_root_mean_squared_error":
            return -float(np.sqrt(mean_squared_error(actual, pred)))
        if self.scoring == "neg_mean_absolute_error":
            return -float(mean_absolute_error(actual, pred))
        return float(r2_score(actual, pred))

    def gridSearchFind(
        self, df_train: pd.DataFrame, param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> SimpleNamespace:
        self._dependency()
        data = self._prepare(df_train, target=True)
        grid = param_grid if param_grid is not None else self._default_param_grid()
        records = []
        for params in self._parameter_candidates(grid):
            scores: List[float] = []
            error = None
            for train_idx, valid_idx in self._windows(len(data)):
                try:
                    fitted = self._fit_model(
                        data.iloc[train_idx][self.target_col].to_numpy(dtype=float), params
                    )
                    pred = self._forecast(fitted, len(valid_idx))
                    actual = data.iloc[valid_idx][self.target_col].to_numpy(dtype=float)
                    scores.append(self._score(actual, pred))
                except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
                    error = str(exc)
                    break
            records.append({
                "params": params,
                "mean_test_score": float(np.mean(scores)) if error is None else -np.inf,
                "split_test_scores": scores,
                "error": error,
            })
        valid = [record for record in records if np.isfinite(record["mean_test_score"])]
        if not valid:
            raise RuntimeError(f"Todas as combinações AutoETS falharam: {records}")
        best = max(valid, key=lambda r: r["mean_test_score"])
        self.best_params = dict(best["params"])
        self.fit_with_best_params(data)
        self.grid_search = SimpleNamespace(
            best_params_=self.best_params.copy(),
            best_score_=best["mean_test_score"],
            best_estimator_=self.best_model,
            cv_results_=records,
        )
        return self.grid_search

    def fit_with_best_params(self, df_train: pd.DataFrame, params_path: Optional[str] = None):
        self._dependency()
        data = self._prepare(df_train, target=True)
        if params_path is not None:
            self.load_best_params(params_path)
        if self.best_params is None:
            raise ValueError("Execute gridSearchFind() ou carregue JSON de parâmetros.")
        self.best_model = self._fit_model(data[self.target_col].to_numpy(dtype=float), self.best_params)
        self.last_train_date = pd.Timestamp(data[self.date_col].iloc[-1])
        return self.best_model

    def fit(self, df_train: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
        """Treino direto sem grid: útil na geração dos OOFs."""
        self.best_params = dict(params or {
            "season_length": self.season_length, "model": "ZZZ", "damped": None,
        })
        return self.fit_with_best_params(df_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model is None or self.last_train_date is None:
            raise ValueError("AutoETS ainda não treinado/carregado.")
        future = self._prepare(df, target=False)
        dates = pd.DatetimeIndex(future[self.date_col])
        first_future_date = self.last_train_date + pd.tseries.frequencies.to_offset(self.freq)
        if dates[0] < first_future_date:
            raise ValueError(
                f"Previsão só pode usar datas a partir de {first_future_date.date()}."
            )
        complete = pd.date_range(first_future_date, dates[-1], freq=self.freq)
        positions = complete.get_indexer(dates)
        if (positions < 0).any():
            raise ValueError(f"Datas de previsão incompatíveis com freq={self.freq!r}.")
        # Faz previsão para TODOS os dias intermediários, devolvendo apenas
        # as datas solicitadas, na ordem cronológica da implementação original.
        full_forecast = self._forecast(self.best_model, len(complete))
        return full_forecast[positions]

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        data = self._prepare(df_test, target=True)
        actual = data[self.target_col].to_numpy(dtype=float)
        pred = self.predict(data)
        denom = np.abs(actual).sum()
        return {
            "mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "r2": float(r2_score(actual, pred)),
            "wape": float(100 * np.abs(actual - pred).sum() / denom) if denom > 0 else float("nan"),
        }

    def save_model(self, filename: str = "autoets_model.joblib") -> Path:
        if self.best_model is None:
            raise ValueError("AutoETS ainda não treinado.")
        path = self.model_dir / filename
        joblib.dump({
            "model": self.best_model,
            "last_train_date": self.last_train_date,
            "target_col": self.target_col,
            "date_col": self.date_col,
            "freq": self.freq,
            "clip_negative": self.clip_negative,
            "missing_dates": self.missing_dates,
        }, path)
        return path

    def load_model(self, model_path: str) -> None:
        self._dependency()
        package = joblib.load(model_path)
        for key in ("target_col", "date_col", "freq", "clip_negative", "missing_dates"):
            if key in package and package[key] != getattr(self, key):
                raise ValueError(f"Configuração '{key}' incompatível com modelo salvo.")
        self.best_model = package["model"]
        self.last_train_date = pd.Timestamp(package["last_train_date"])

    def save_best_params(self, filename: str = "autoets_best_params.json") -> Path:
        if self.best_params is None:
            raise ValueError("Não há parâmetros selecionados.")
        path = self.model_dir / filename
        path.write_text(json.dumps({
            "best_params": self.best_params,
            "best_score": self.grid_search.best_score_ if self.grid_search else None,
            "scoring": self.scoring,
            "n_splits": self.n_splits,
            "horizon": self.horizon,
            "target_col": self.target_col,
            "date_col": self.date_col,
            "freq": self.freq,
            "missing_dates": self.missing_dates,
            "cv_strategy": "expanding_window_non_overlapping_horizon",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_best_params(self, params_path: str) -> Dict[str, Any]:
        data = json.loads(Path(params_path).read_text(encoding="utf-8"))
        for key in ("target_col", "date_col", "freq", "missing_dates"):
            if key in data and data[key] != getattr(self, key):
                raise ValueError(f"Parâmetro '{key}' incompatível com JSON.")
        params = data.get("best_params")
        if not isinstance(params, dict) or not params:
            raise ValueError("JSON sem best_params válido.")
        self.best_params = params
        return params

    def save_metrics(self, metrics: Dict[str, float], filename: str = "autoets_metrics.json") -> Path:
        path = self.model_dir / filename
        cleaned = {k: (float(v) if np.isfinite(v) else None) for k, v in metrics.items()}
        path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
        return path

    def get_model(self):
        if self.best_model is None:
            raise ValueError("Modelo não treinado.")
        return self.best_model

    def get_model_type(self):
        return type(self.get_model())

    @staticmethod
    def promote_model_to_production(
        model_source_path: str, production_dir: str, model_version: str,
        target: str = "quantity_sum", training_period: Optional[str] = None,
    ) -> Dict[str, str]:
        dest = Path(production_dir)
        dest.mkdir(parents=True, exist_ok=True)
        artifact = dest / f"autoets_model_{model_version}.joblib"
        shutil.copy2(model_source_path, artifact)
        metadata = dest / f"autoets_metadata_{model_version}.json"
        metadata.write_text(json.dumps({
            "solution_name": "PharmaDemand AutoETS", "version": model_version,
            "target": target, "training_period": training_period,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact": artifact.name,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"model": str(artifact), "metadata": str(metadata)}
