"""Ridge metamodelo para stacking temporal do PharmaDemand.

Entrada: previsoes OOF dos modelos-base (treinamento) e previsoes de
holdout de modelos-base treinados apenas em dados anteriores ao holdout.
Nao passe features reais da data-alvo nem previsoes in-sample dos modelos-base.

Exemplo:
    model = RidgeStackingRegressor(
        target_col='quantity',
        numeric_features=['pred_lightgbm', 'pred_xgboost',
                          'pred_sarimax', 'pred_autotheta'],
        categorical_features=['category'],
        date_col='date', gap_days=15, n_splits=3,
        model_dir='models/stacking',
    )
    model.gridSearchFind(df_oof_train)
    model.evaluate(df_holdout)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RidgeStackingRegressor:
    """Metamodelo Ridge com GridSearchCV, CV temporal e API similar ao LightGBM.

    ``date_col`` deve representar a data de origem da previsao (forecast
    origin). ``gap_days`` retira do treino da CV os ultimos dias anteriores
    ao inicio da validacao. Se os labels de D+15 so ficam disponiveis
    15 dias depois da origem, use gap_days >= 15.

    Requer que as colunas ``pred_*`` do treino sejam previsoes OOF genuinas.
    Este requisito nao pode ser garantido apenas inspecionando o DataFrame.
    """

    def __init__(
        self,
        target_col: str,
        numeric_features: List[str],
        categorical_features: Optional[List[str]] = None,
        model_dir: Union[str, Path] = "models/stacking",
        random_state: int = 42,
        n_splits: int = 3,
        scoring: str = "neg_root_mean_squared_error",
        date_col: str = "date",
        gap_days: int = 15,
        require_prediction_features: bool = True,
    ) -> None:
        self.target_col = target_col
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features or [])
        self.model_dir = Path(model_dir)
        self.random_state = random_state  # interface compativel; Ridge e deterministico
        self.n_splits = n_splits
        self.scoring = scoring
        self.date_col = date_col
        self.gap_days = gap_days
        self.require_prediction_features = require_prediction_features
        self.pipeline: Optional[Pipeline] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.best_model: Optional[Pipeline] = None
        self.best_params: Optional[Dict[str, Any]] = None

        if not self.numeric_features:
            raise ValueError("Informe ao menos uma coluna numerica de previsao pred_*.")
        if len(set(self.numeric_features + self.categorical_features)) != (
            len(self.numeric_features) + len(self.categorical_features)
        ):
            raise ValueError("Ha features duplicadas entre numericas e categoricas.")
        if self.target_col in self.numeric_features + self.categorical_features:
            raise ValueError("O target nao pode ser feature do metamodelo.")
        if self.date_col in self.numeric_features + self.categorical_features:
            raise ValueError("A data de origem deve ser chave temporal, nao feature.")
        if require_prediction_features and any(
            not col.startswith("pred_") for col in self.numeric_features
        ):
            raise ValueError(
                "No stacking, numeric_features devem conter apenas previsoes "
                "dos modelos-base, nomeadas pred_*. Para liberar covariaveis "
                "adicionais, defina require_prediction_features=False."
            )
        if n_splits < 2:
            raise ValueError("n_splits deve ser >= 2.")
        if not isinstance(gap_days, int) or gap_days < 0:
            raise ValueError("gap_days deve ser um inteiro >= 0.")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # sklearn antigo
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers = [("numeric", numeric_transformer, self.numeric_features)]
        if self.categorical_features:
            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._encoder()),
            ])
            transformers.append(("categorical", categorical_transformer,
                                 self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop",
                                 sparse_threshold=0.0)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("preprocessor", self._build_preprocessor()),
            ("model", Ridge(alpha=1.0, fit_intercept=True)),
        ])

    def _default_param_grid(self) -> Dict[str, List[Any]]:
        return {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "model__fit_intercept": [True],
        }

    def _validate_dataframe(self, df: pd.DataFrame, require_target: bool = True) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("O DataFrame deve conter pelo menos uma linha.")
        required = self.numeric_features + self.categorical_features + [self.date_col]
        if require_target:
            required.append(self.target_col)
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Colunas obrigatorias ausentes: {missing}")
        dates = pd.to_datetime(df[self.date_col], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"{self.date_col} tem datas ausentes ou invalidas.")
        for col in self.numeric_features:
            values = pd.to_numeric(df[col], errors="coerce")
            # No metamodelo, nao imputar uma previsao inteira ausente.
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"Feature de previsao {col!r} contem NaN/inf/valor invalido.")
        if require_target:
            values = pd.to_numeric(df[self.target_col], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"Target {self.target_col!r} contem valores invalidos.")

    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data[self.date_col] = pd.to_datetime(data[self.date_col])
        return data.sort_values(self.date_col, kind="stable").reset_index(drop=True)

    def _date_based_cv_splits(self, df_sorted: pd.DataFrame):
        """Expanding window por DATA unica, com embargo temporal em dias.

        Cada fold usa somente datas anteriores a validacao. Se a disponibilidade
        dos rotulos for posterior a D+gap_days, ampliar gap_days.
        """
        dates = pd.to_datetime(df_sorted[self.date_col])
        unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
        if len(unique_dates) <= self.n_splits:
            raise ValueError(
                f"CV insuficiente: {len(unique_dates)} datas unicas para "
                f"n_splits={self.n_splits}."
            )
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits = []
        for fold, (train_d, valid_d) in enumerate(tscv.split(unique_dates), start=1):
            first_valid = unique_dates[valid_d[0]]
            # Conservador: origens de treino estritamente anteriores ao
            # primeiro dia de validacao menos o horizonte de disponibilidade.
            cutoff = first_valid - pd.Timedelta(days=self.gap_days)
            eligible_dates = unique_dates[train_d]
            eligible_dates = eligible_dates[eligible_dates < cutoff]
            train_idx = np.flatnonzero(dates.isin(eligible_dates).to_numpy())
            valid_idx = np.flatnonzero(dates.isin(unique_dates[valid_d]).to_numpy())
            if len(train_idx) < 2 or len(valid_idx) < 2:
                raise ValueError(
                    f"Fold {fold} sem amostras suficientes apos gap_days={self.gap_days}. "
                    "Reduza n_splits, ajuste a janela ou revise a data de origem."
                )
            splits.append((train_idx, valid_idx))
        return splits

    def gridSearchFind(
        self, df_train: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> GridSearchCV:
        """Ajusta Ridge SOMENTE sobre previsoes OOF dos modelos-base."""
        self._validate_dataframe(df_train)
        data = self._sort_by_date(df_train)
        X = data[self.numeric_features + self.categorical_features]
        y = data[self.target_col]
        self.pipeline = self._build_pipeline()
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid or self._default_param_grid(),
            scoring=self.scoring,
            cv=self._date_based_cv_splits(data),
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            refit=True,
            error_score="raise",
        )
        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_
        self.best_params = dict(self.grid_search.best_params_)
        return self.grid_search

    def load_best_params(self, params_path: Union[str, Path]) -> Dict[str, Any]:
        with open(params_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.best_params = dict(payload.get("best_params", payload))
        if not self.best_params:
            raise ValueError("JSON sem best_params.")
        return self.best_params

    def fit_with_best_params(
        self, df_train: pd.DataFrame,
        params_path: Optional[Union[str, Path]] = None,
    ) -> Pipeline:
        self._validate_dataframe(df_train)
        data = self._sort_by_date(df_train)
        if params_path is not None:
            self.load_best_params(params_path)
        if self.best_params is None:
            raise ValueError("Execute gridSearchFind() ou load_best_params() antes.")
        self.pipeline = self._build_pipeline()
        self.pipeline.set_params(**self.best_params)
        self.pipeline.fit(
            data[self.numeric_features + self.categorical_features],
            data[self.target_col],
        )
        self.best_model = self.pipeline
        return self.best_model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Metamodelo ainda nao treinado/carregado.")
        self._validate_dataframe(df, require_target=False)
        X = df[self.numeric_features + self.categorical_features]
        return np.asarray(self.best_model.predict(X), dtype=float)

    @staticmethod
    def calculate_metrics(y_true, y_pred) -> Dict[str, Optional[float]]:
        """WAPE em %; bias positivo significa superprevisao."""
        actual = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        if len(actual) != len(pred) or len(actual) == 0:
            raise ValueError("y_true e y_pred devem ter comprimentos iguais nao vazios.")
        abs_total = np.abs(actual).sum()
        return {
            "mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "r2": float(r2_score(actual, pred)) if len(actual) > 1 else None,
            "wape_pct": float(100 * np.abs(actual - pred).sum() / abs_total)
            if abs_total > 0 else None,
            "bias": float(np.mean(pred - actual)),
        }

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Optional[float]]:
        self._validate_dataframe(df_test)
        return self.calculate_metrics(df_test[self.target_col], self.predict(df_test))

    def get_model(self) -> Ridge:
        if self.best_model is None:
            raise ValueError("Metamodelo ainda nao treinado/carregado.")
        return self.best_model.named_steps["model"]

    def get_model_type(self):
        return type(self.get_model())

    def get_coefficients(self) -> pd.DataFrame:
        """Coeficientes padronizados e equivalentes em escala original.

        Para features numericas, raw_coefficient = coefficient / std_da_coluna.
        Para categorias one-hot, nao ha padronizacao. Coeficientes
        de modelos-base muito correlacionados podem ser instaveis.
        """
        if self.best_model is None:
            raise ValueError("Metamodelo ainda nao treinado/carregado.")
        prep = self.best_model.named_steps["preprocessor"]
        names = prep.get_feature_names_out()
        coefs = np.asarray(self.get_model().coef_, dtype=float).reshape(-1)
        if len(names) != len(coefs):
            raise RuntimeError("Coeficientes e nomes das features nao correspondem.")
        original = coefs.copy()
        std = prep.named_transformers_["numeric"].named_steps["scaler"].scale_
        original[:len(self.numeric_features)] /= std
        return pd.DataFrame({
            "feature": names,
            "coefficient": coefs,
            "raw_coefficient": original,
        })

    def save_model(self, filename: str = "ridge_stacking_model.joblib") -> Path:
        if self.best_model is None:
            raise ValueError("Metamodelo ainda nao treinado/carregado.")
        path = self.model_dir / filename
        joblib.dump(self.best_model, path)
        return path

    def load_model(self, model_path: Union[str, Path]) -> None:
        self.best_model = joblib.load(model_path)
        if not isinstance(self.best_model, Pipeline):
            raise TypeError("Arquivo joblib nao contem um Pipeline do sklearn.")

    def save_best_params(self, filename: str = "ridge_stacking_best_params.json") -> Path:
        if self.best_params is None:
            raise ValueError("Execute gridSearchFind() ou carregue parametros.")
        payload = {
            "best_params": self.best_params,
            "best_score": float(self.grid_search.best_score_)
            if self.grid_search is not None else None,
            "scoring": self.scoring,
            "n_splits": self.n_splits,
            "gap_days": self.gap_days,
            "target_col": self.target_col,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "date_col": self.date_col,
            "cv_strategy": "expanding_window_by_unique_forecast_origin_with_gap",
            "source_expectation": "OOF predictions of base models (never in-sample)",
        }
        path = self.model_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        return path

    def save_metrics(
        self, metrics: Dict[str, Optional[float]],
        filename: str = "ridge_stacking_metrics.json",
    ) -> Path:
        path = self.model_dir / filename
        clean = {k: float(v) if v is not None else None for k, v in metrics.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False, allow_nan=False)
        return path
