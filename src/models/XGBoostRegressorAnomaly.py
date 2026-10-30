"""XGBoost para regressão temporal com features externas de anomalia.

Depende de sort_by_forecast_date e build_date_based_cv_splits do PharmaDemand.
A classe NÃO treina o Isolation Forest; apenas utiliza suas features quando
presentes em numeric_features/categorical_features.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.features.features import build_date_based_cv_splits, sort_by_forecast_date


class XGBoostRegressorAnomaly:
    """Pipeline XGBoost e GridSearchCV usando validação cruzada temporal.

    A validação temporal depende de build_date_based_cv_splits (não fornecida).
    Features do Isolation Forest devem ser previamente calculadas sem vazamento
    temporal, de acordo com o horizonte de previsão.
    """

    def __init__(
        self,
        target_col: str,
        numeric_features: List[str],
        categorical_features: Optional[List[str]] = None,
        model_dir: str = "models/baseline",
        random_state: int = 42,
        n_splits: int = 3,
        scoring: str = "neg_root_mean_squared_error",
        date_col: str = "date",
        grid_n_jobs: int = 2,
        model_n_jobs: int = 1,
    ):
        self.target_col = target_col
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features or [])
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        self.n_splits = n_splits
        self.scoring = scoring
        self.date_col = date_col
        # Evita n_jobs=-1 dentro de n_jobs=-1, que pode sobrecarregar CPU/RAM.
        self.grid_n_jobs = grid_n_jobs
        self.model_n_jobs = model_n_jobs

        features = self.numeric_features + self.categorical_features
        if not features:
            raise ValueError("Informe ao menos uma feature.")
        if len(set(features)) != len(features):
            raise ValueError("Há features repetidas nas listas numéricas/categóricas.")
        if self.target_col in features:
            raise ValueError("Target não pode constar na lista de features (leakage).")
        if self.date_col in features:
            raise ValueError(
                "A coluna de data bruta não deve ser usada diretamente como feature. "
                "Use colunas de calendário derivadas (dia da semana, mês etc.)."
            )
        if n_splits < 2:
            raise ValueError("n_splits deve ser >= 2.")

        self.pipeline: Optional[Pipeline] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.best_model: Optional[Pipeline] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _create_one_hot_encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:  # compatibilidade com scikit-learn antigo
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _build_preprocessor(self) -> ColumnTransformer:
        transformers = []
        if self.numeric_features:
            transformers.append((
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                self.numeric_features,
            ))
        if self.categorical_features:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", self._create_one_hot_encoder()),
                ]),
                self.categorical_features,
            ))
        return ColumnTransformer(transformers, remainder="drop")

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("preprocessor", self._build_preprocessor()),
            ("model", XGBRegressor(
                objective="reg:squarederror",  # XGBoost (não 'regression')
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=self.model_n_jobs,
                verbosity=0,  # XGBoost: somente 0, 1, 2 ou 3
            )),
        ])

    def _default_param_grid(self) -> Dict[str, List[Any]]:
        # 2*2*3*2*1*1*2 = 48 combinações; 3 folds = 144 fits + refit.
        return {
            "model__n_estimators": [300, 600],
            "model__learning_rate": [0.03, 0.07],
            "model__max_depth": [3, 5, 7],
            "model__min_child_weight": [1, 5],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
            "model__reg_lambda": [1.0, 5.0],
        }

    def _validate_dataframe(self, df: pd.DataFrame, require_date: bool = False) -> None:
        required = self.numeric_features + self.categorical_features + [self.target_col]
        if require_date:
            required.append(self.date_col)
        missing = [name for name in required if name not in df.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
        if df.empty:
            raise ValueError("DataFrame está vazio.")
        if df[self.target_col].isna().any():
            raise ValueError(f"Target {self.target_col!r} contém NaN.")
        if not pd.api.types.is_numeric_dtype(df[self.target_col]):
            raise ValueError(f"Target {self.target_col!r} deve ser numérico.")
        if not np.isfinite(df[self.target_col].to_numpy(dtype=float)).all():
            raise ValueError(f"Target {self.target_col!r} contém valores não finitos.")

    def gridSearchFind(
        self,
        df_train: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> GridSearchCV:
        """Executa busca de hiperparâmetros com folds cronológicos."""
        self._validate_dataframe(df_train, require_date=True)
        df_train = sort_by_forecast_date(df_train, date_col=self.date_col)
        X_train = df_train[self.numeric_features + self.categorical_features].copy()
        y_train = df_train[self.target_col].copy()

        cv = build_date_based_cv_splits(
            df_train, n_splits=self.n_splits, date_col=self.date_col
        )
        self.pipeline = self._build_pipeline()
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid if param_grid is not None else self._default_param_grid(),
            scoring=self.scoring,
            cv=cv,
            n_jobs=self.grid_n_jobs,
            verbose=1,
            return_train_score=True,   # mantém métricas de treino para análise de overfitting
            error_score="raise",       # apresenta a exceção real imediatamente
            pre_dispatch="2*n_jobs",
            refit=True,
        )
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        return self.grid_search

    def fit_with_best_params(
        self,
        df_train: pd.DataFrame,
        params_path: Optional[str] = None,
    ) -> Pipeline:
        """Treina com parâmetros obtidos em uma busca anterior (ou JSON)."""
        self._validate_dataframe(df_train, require_date=True)
        df_train = sort_by_forecast_date(df_train, date_col=self.date_col)
        if params_path is not None:
            self.load_best_params(params_path)
        if self.best_params is None:
            raise ValueError("Execute gridSearchFind ou informe params_path primeiro.")

        self.pipeline = self._build_pipeline()
        self.pipeline.set_params(**self.best_params)
        self.pipeline.fit(
            df_train[self.numeric_features + self.categorical_features].copy(),
            df_train[self.target_col].copy(),
        )
        self.best_model = self.pipeline
        return self.best_model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Modelo ainda não treinado/carregado.")
        missing = [name for name in self.numeric_features + self.categorical_features if name not in df]
        if missing:
            raise ValueError(f"Colunas faltando para predição: {missing}")
        X = df[self.numeric_features + self.categorical_features].copy()
        return self.best_model.predict(X)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        self._validate_dataframe(df_test)
        y_true = df_test[self.target_col].to_numpy(dtype=float)
        y_pred = self.predict(df_test)
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }
        return metrics

    def save_model(self, filename: str = "xgboost_anomalies_model.joblib") -> Path:
        if self.best_model is None:
            raise ValueError("Modelo ainda não treinado/carregado.")
        path = self.model_dir / filename
        joblib.dump(self.best_model, path)
        return path

    def save_best_params(self, filename: str = "xgboost_anomalies_best_params.json") -> Path:
        if self.best_params is None:
            raise ValueError("Nenhum conjunto de hiperparâmetros disponível.")
        data = {
            "best_params": self.best_params,
            "best_score": (
                float(self.grid_search.best_score_)
                if self.grid_search is not None and hasattr(self.grid_search, "best_score_")
                else None
            ),
            "scoring": self.scoring,
            "n_splits": self.n_splits,
            "target_col": self.target_col,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "date_col": self.date_col,
            "cv_strategy": "expanding_window_by_unique_date",
        }
        path = self.model_dir / filename
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        return path

    def load_best_params(self, params_path: str) -> Dict[str, Any]:
        """Carrega hiperparâmetros de JSON gerado por save_best_params."""
        with Path(params_path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        params = data.get("best_params")
        if not isinstance(params, dict) or not params:
            raise ValueError("Arquivo JSON não contém best_params válido.")
        if "target_col" in data and data["target_col"] != self.target_col:
            raise ValueError("Target do JSON não coincide com o target atual.")
        for key in ("numeric_features", "categorical_features", "date_col"):
            if key in data and data[key] != getattr(self, key):
                raise ValueError(f"Configuração '{key}' do JSON difere da classe atual.")
        self.best_params = params
        return params

    def save_metrics(
        self,
        metrics: Dict[str, float],
        filename: str = "xgboost_anomalies_metrics.json",
    ) -> Path:
        path = self.model_dir / filename
        with path.open("w", encoding="utf-8") as file:
            json.dump({k: float(v) for k, v in metrics.items()}, file, indent=4)
        return path

    def load_model(self, model_path: str) -> None:
        """Carrega pipeline e recupera listas de features quando disponíveis."""
        loaded = joblib.load(model_path)
        if not isinstance(loaded, Pipeline):
            raise TypeError("O arquivo joblib deve conter um Pipeline.")
        self.best_model = loaded
        # As listas informadas no construtor precisam coincidir com as do modelo.
        preprocessor = loaded.named_steps["preprocessor"]
        for name, _, features in preprocessor.transformers:
            if name == "numeric" and list(features) != self.numeric_features:
                raise ValueError("numeric_features divergem das utilizadas no modelo salvo.")
            if name == "categorical" and list(features) != self.categorical_features:
                raise ValueError("categorical_features divergem das utilizadas no modelo salvo.")

    def get_model(self) -> XGBRegressor:
        if self.best_model is None:
            raise ValueError("Modelo ainda não treinado/carregado.")
        return self.best_model.named_steps["model"]

    def get_model_type(self):
        return type(self.get_model())

    @staticmethod
    def promote_model_to_production(
        lgbm_source_path: str,  # nome legado, mantido para compatibilidade
        isolation_forest_source_path: str,
        production_dir: str,
        model_version: str,
        target: str = "quantity_sum",
        training_period: Optional[str] = None,
        test_period: Optional[str] = None,
    ) -> Dict[str, str]:
        """Copia XGBoost e Isolation Forest já treinados e grava metadados.

        OBS: 'lgbm_source_path' é nome legado do método original; o arquivo
        fornecido deve conter um pipeline XGBoost, não um LightGBM.
        """
        production_path = Path(production_dir)
        production_path.mkdir(parents=True, exist_ok=True)
        xgb_name = f"xgboost_model_{model_version}.joblib"
        if_name = f"isolation_forest_{model_version}.joblib"
        metadata_name = f"model_metadata_{model_version}.json"
        xgb_path = production_path / xgb_name
        if_path = production_path / if_name
        metadata_path = production_path / metadata_name
        shutil.copy2(lgbm_source_path, xgb_path)
        shutil.copy2(isolation_forest_source_path, if_path)
        metadata = {
            "solution_name": "PharmaDemand - XGBoost + Isolation Forest",
            "solution_version": model_version,
            "problem_type": "Forecasting de demanda com features de anomalia",
            "target": target,
            "training_period": training_period,
            "test_period": test_period,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "supervised_model": xgb_name,
                "anomaly_model": if_name,
                "metadata": metadata_name,
            },
            "ml_model": {
                "name": "XGBoost Regressor",
                "role": "Previsão de demanda a partir de features disponíveis na data da previsão.",
            },
            "anomaly_model": {
                "name": "Isolation Forest",
                "role": "Features de anomalia calculadas externamente sem informação futura.",
            },
        }
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)
        return {
            "production_dir": str(production_path),
            "xgboost_model": str(xgb_path),
            "isolation_forest": str(if_path),
            "metadata": str(metadata_path),
        }
