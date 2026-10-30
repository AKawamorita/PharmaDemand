import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

# from src.models.forecast_features import (
#     assert_leakage_safe_features,
#     build_date_based_cv_splits,
#     sort_by_forecast_date,
# )

from src.features.features import (sort_by_forecast_date, build_date_based_cv_splits)

class LightGBMRegressorAnomaly:
    """
    Baseline model using LightGBM Regressor.

    This class trains a LightGBM regression model with anomaly detection features.
    It is useful as a baseline to compare against more advanced solutions, such as
    Isolation Forest + LightGBM.

    The class supports:
    - Numeric and categorical feature preprocessing
    - Expanding-window temporal cross-validation by unique date
    - GridSearchCV for hyperparameter tuning
    - Model saving with joblib
    - Best hyperparameters saving as JSON

    Parameters
    ----------
    target_col : str
        Name of the target column to predict.

    numeric_features : List[str]
        List of numeric columns used by the model.

    categorical_features : Optional[List[str]]
        List of categorical columns used by the model.

    model_dir : str
        Directory where the trained model and metadata will be saved.

    random_state : int
        Random seed for reproducibility.

    n_splits : int
        Number of expanding temporal folds.

    scoring : str
        Scoring metric used by GridSearchCV.
        Example: "neg_root_mean_squared_error", "neg_mean_absolute_error", "r2".
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
        date_col: str = "date"
    ):
        self.target_col = target_col
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features or []
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        self.n_splits = n_splits
        self.scoring = scoring
        self.date_col = date_col

        self.pipeline = None
        self.grid_search = None
        self.best_model = None
        self.best_params = None

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _create_one_hot_encoder(self):
        """
        Create a OneHotEncoder compatible with different scikit-learn versions.

        Returns
        -------
        OneHotEncoder
            Configured encoder for categorical variables.
        """
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Build the preprocessing pipeline.

        Numeric features are filled with the median.
        Categorical features are filled with the most frequent value and encoded
        using OneHotEncoder.

        Returns
        -------
        ColumnTransformer
            Preprocessor for numeric and categorical features.
        """
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._create_one_hot_encoder())
            ]
        )

        transformers = []

        if self.numeric_features:
            transformers.append(
                ("numeric", numeric_transformer, self.numeric_features)
            )

        if self.categorical_features:
            transformers.append(
                ("categorical", categorical_transformer, self.categorical_features)
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        return preprocessor

    def _build_pipeline(self) -> Pipeline:
        """
        Build the full machine learning pipeline.

        The pipeline includes preprocessing and the LightGBM Regressor model.

        Returns
        -------
        Pipeline
            Complete training pipeline.
        """
        preprocessor = self._build_preprocessor()

        model = LGBMRegressor(
            objective="regression",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True #evita que o LightGBM perca tempo testando automaticamente a melhor estratégia interna
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        return pipeline

    def _default_param_grid(self) -> Dict[str, List[Any]]:
        """
        Create a default hyperparameter grid for LightGBM.

        The grid is intentionally moderate to avoid excessive training time.
        You can expand it later if needed.

        Returns
        -------
        Dict[str, List[Any]]
            Hyperparameter grid used by GridSearchCV.
        """
        return {
        "model__n_estimators": [500, 1000],
        "model__learning_rate": [0.03, 0.07],
        "model__num_leaves": [15, 31],
        "model__max_depth": [5, 8, 12],
        "model__min_child_samples": [20, 50],
        "model__subsample": [0.8],
        "model__subsample_freq": [1],
        "model__colsample_bytree": [0.8],
        "model__reg_lambda": [1e-8, 10.0]
    }

    def gridSearchFind(
        self,
        df_train: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> GridSearchCV:
        """
        Train the LightGBM baseline model using GridSearchCV.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training dataframe containing features and target.

        param_grid : Optional[Dict[str, List[Any]]]
            Custom hyperparameter grid. If None, a default grid is used.

        Returns
        -------
        GridSearchCV
            Fitted GridSearchCV object.
        """
        self._validate_dataframe(df_train)
        # assert_leakage_safe_features(
        #     self.numeric_features + self.categorical_features,
        #     target_col=self.target_col,
        # )
        df_train = sort_by_forecast_date(df_train, date_col=self.date_col)

        X_train = df_train[self.numeric_features + self.categorical_features].copy()
        y_train = df_train[self.target_col].copy()

        self.pipeline = self._build_pipeline()

        cv = build_date_based_cv_splits(
            df_train,
            n_splits=self.n_splits,
            date_col=self.date_col,
        )

        if param_grid is None:
            param_grid = self._default_param_grid()

        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=cv,
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )

        self.grid_search.fit(X_train, y_train)

        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_        

        return self.grid_search
    
    def fit_with_best_params(
        self,
        df_train: pd.DataFrame,
        params_path: Optional[str] = None
    ) -> Pipeline:
        """
        Train the LightGBM model using previously saved best hyperparameters.

        This method is useful when GridSearchCV was already executed before.
        It avoids running the full hyperparameter search again.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training dataframe containing features and target.

        params_path : Optional[str]
            Path to the JSON file with the best hyperparameters.
            If None, the method uses self.best_params.

        Returns
        -------
        Pipeline
            Trained pipeline using the loaded best hyperparameters.
        """
        self._validate_dataframe(df_train)
        # assert_leakage_safe_features(
        #     self.numeric_features + self.categorical_features,
        #     target_col=self.target_col,
        # )
        df_train = sort_by_forecast_date(df_train, date_col=self.date_col)

        if params_path is not None:
            self.load_best_params(params_path)

        if self.best_params is None:
            raise ValueError(
                "Best parameters are not available. "
                "Please call load_best_params() or provide params_path."
            )

        X_train = df_train[self.numeric_features + self.categorical_features].copy()
        y_train = df_train[self.target_col].copy()

        self.pipeline = self._build_pipeline()

        # Apply parameters with the pipeline prefix, for example:
        # model__n_estimators, model__learning_rate, etc.
        self.pipeline.set_params(**self.best_params)

        self.pipeline.fit(X_train, y_train)

        self.best_model = self.pipeline

        return self.best_model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the best trained model.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the input features.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.best_model is None:
            raise ValueError("Model is not trained yet. Please call fit() first.")

        X = df[self.numeric_features + self.categorical_features].copy()
        return self.best_model.predict(X)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the trained model on a test dataset.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test dataframe containing features and target.

        Returns
        -------
        Dict[str, float]
            Dictionary with MAE, RMSE and R2 metrics.
        """
        self._validate_dataframe(df_test)

        y_true = df_test[self.target_col].copy()
        y_pred = self.predict(df_test)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

        return metrics

    def save_model(self, filename: str = "lightgbm_anomalies_model.joblib") -> Path:
        """
        Save the trained model pipeline as a joblib file.

        Parameters
        ----------
        filename : str
            Name of the model file.

        Returns
        -------
        Path
            Full path of the saved model.
        """
        if self.best_model is None:
            raise ValueError("Model is not trained yet. Please call fit() first.")

        model_path = self.model_dir / filename
        joblib.dump(self.best_model, model_path)

        return model_path

    def save_best_params(self, filename: str = "lightgbm_anomalies_best_params.json") -> Path:
        """
        Save the best hyperparameters found by GridSearchCV.

        Parameters
        ----------
        filename : str
            Name of the JSON file.

        Returns
        -------
        Path
            Full path of the saved JSON file.
        """
        if self.best_params is None:
            raise ValueError("Best parameters are not available. Please call fit() first.")

        params_path = self.model_dir / filename

        data = {
            "best_params": self.best_params,
            "best_score": self.grid_search.best_score_,
            "scoring": self.scoring,
            "n_splits": self.n_splits,
            "target_col": self.target_col,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "date_col": self.date_col,
            "cv_strategy": "expanding_window_by_unique_date"
        }

        with open(params_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        return params_path

    def save_metrics(
        self,
        metrics: Dict[str, float],
        filename: str = "lightgbm_anomalies_metrics.json"
    ) -> Path:
        """
        Save model evaluation metrics as JSON.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary with evaluation metrics.

        filename : str
            Name of the metrics JSON file.

        Returns
        -------
        Path
            Full path of the saved metrics file.
        """
        metrics_path = self.model_dir / filename

        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        return metrics_path

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model pipeline from disk.

        Parameters
        ----------
        model_path : str
            Path of the saved joblib model.
        """        
        self.best_model = joblib.load(model_path)

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate if the dataframe contains all required columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate.

        Raises
        ------
        ValueError
            If one or more required columns are missing.
        """
        required_cols = self.numeric_features + self.categorical_features + [self.target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        

    def get_model(self):
        """
        Return the internal tree model used for prediction.
        If the saved object is a Pipeline, return its final estimator.
        """
        if self.best_model is None:
            raise ValueError("The internal model is not trained or was not loaded.")

        if isinstance(self.best_model, Pipeline):
            return self.best_model.steps[-1][1]

        return self.best_model

    def get_model_type(self):
        """
        Return the type of the internal trained model.
        """
        return type(self.get_model())
  
    def promote_model_to_production(
        lgbm_source_path: str,
        isolation_forest_source_path: str,
        production_dir: str,
        model_version: str,
        target: str = "quantity_sum",
        training_period: str = "2021-2022",
        test_period: str = "2023",
    ):
        """
        Copies trained model artifacts to the production folder and creates
        a metadata JSON file describing the complete inference solution.
        """

        production_path = Path(production_dir)
        production_path.mkdir(parents=True, exist_ok=True)

        lgbm_dest_name = f"lgbm_model_{model_version}.joblib"
        if_dest_name = f"isolation_forest_{model_version}.joblib"
        metadata_dest_name = f"model_metadata_{model_version}.json"

        lgbm_dest_path = production_path / lgbm_dest_name
        if_dest_path = production_path / if_dest_name
        metadata_dest_path = production_path / metadata_dest_name

        shutil.copy2(lgbm_source_path, lgbm_dest_path)
        shutil.copy2(isolation_forest_source_path, if_dest_path)

        metadata = {
            "solution_name": "Beverage Sales Hybrid Forecasting Model",
            "solution_version": model_version,
            "problem_type": "Demand forecasting with anomaly-aware features",
            "target": target,
            "training_period": training_period,
            "test_period": test_period,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "benchmark_notebook",

            "artifacts": {
                "supervised_model": lgbm_dest_name,
                "anomaly_model": if_dest_name,
                "metadata": metadata_dest_name
            },

            "baseline_model": {
                "name": "LightGBM Regressor",
                "description": "Baseline supervised model trained without anomaly features."
            },

            "hybrid_model": {
                "name": "Isolation Forest + LightGBM Regressor",
                "description": (
                    "Final model using one-day-lagged anomaly score and flag "
                    "as causal forecasting features."
                )
            },

            "anomaly_model": {
                "name": "Isolation Forest",
                "role": "Detects unusual sales behavior and generates anomaly features used by the final supervised model.",
                "output_features": [
                    "anomaly_score",
                    "anomaly_flag"
                ]
            },

            "ml_model": {
                "name": "LightGBM Regressor",
                "role": (
                    "Predicts daily beverage demand using known calendar fields, "
                    "past-only rolling history, and lagged anomaly signals."
                )
            },

            "notes": [
                "The API project should load these artifacts instead of retraining the models.",
                "This metadata file is used for traceability, model versioning, and the /model-info endpoint.",
                "Forecast date t never consumes realized business or anomaly fields from t."
            ]
        }

        with open(metadata_dest_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

        return {
            "production_dir": str(production_path),
            "lgbm_model": str(lgbm_dest_path),
            "isolation_forest": str(if_dest_path),
            "metadata": str(metadata_dest_path)
        }
