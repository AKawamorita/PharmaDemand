# load_parket
# fit_model
# autotest (teste de integridade, se nao precisa deixa em branco)
# find_hyper_parameters
# train (treino)
# evaluate  treino

# save_best_params
# load_params

# save_model

"""PharmaDemand | SARIMAX por categoria e horizonte, respeitando o split do Parquet.

Salvar este arquivo em ``src/models/train_sarimax.py``.

Executar na raiz do projeto (dependências: pandas, pyarrow, statsmodels, numpy):
    python -m src.models.train_sarimax \
        --parquet data/features/anomaly_predictions.parquet \
        --category N02BE --horizon-days 15

Interface Python:
    from src.models.train_sarimax import SARIMAXRegressor, load_pharma_splits
    train, test = load_pharma_splits('data/features/anomaly_predictions.parquet', 'N02BE', 15)
    model = SARIMAXRegressor(category='N02BE', horizon_days=15)
    model.fit(train)
    scores, forecasts = model.evaluate(test)
    model.save('models/baseline/sarimax/N02BE/h15')

IMPORTANTE:
- ``date`` = data do ALVO; ``forecast_origin`` = data do conhecimento disponível.
- Modela exclusivamente ``quantity`` (sem features de lag e sem anomaly_score).
- Avaliação de passo EXATO h só começa quando forecast_origin >= fim do treino.
  Isso exclui os h-1 primeiros dias de teste; não usa informações futuras.
- Para pontuar TODOS os dias do teste, há também backtest em blocos (passos 1..h),
  que NÃO é equivalente à predição supervisionada sempre em D+h.
- Treino, teste, transformações e previsões são processados em memória.
- Não gera CSVs nem regrava os datasets de treino e teste.
- Por padrão, persiste apenas o modelo (.pkl) e as métricas/metadados (.json).
  Previsões são DataFrames e só são salvas (em um Parquet único) se solicitado.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from src.config.config import MODELS_PARAM,DATA_RAW,DATA_FEATURES,DATA_EXPORTS,DATA_PROCESSED
from src.repository.parquet_repository import ParquetRepository
parket_data = ParquetRepository(DATA_FEATURES)

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 0, 7)
TREND = "n"
ALLOWED_HORIZONS = (7, 15)


def calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Exógenas conhecidas para qualquer data, inclusive no futuro."""
    idx = pd.DatetimeIndex(index)
    day = idx.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "is_weekend": (idx.dayofweek >= 5).astype(float),
            "annual_sin": np.sin(2 * np.pi * day / 365.25),
            "annual_cos": np.cos(2 * np.pi * day / 365.25),
        },
        index=idx,
    )


def quality_metrics(y_true, y_pred) -> dict:
    """Métricas no conjunto efetivamente previsto."""
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    if not len(y) or len(y) != len(yhat) or not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError("As séries de métricas devem ter mesmo tamanho, sem NaN/Inf.")
    delta = np.abs(y - yhat)
    sm_den = np.abs(y) + np.abs(yhat)
    wape_den = np.abs(y).sum()
    ss_total = np.square(y - y.mean()).sum()
    return {
        "n_predictions": int(len(y)),
        "mae": float(delta.mean()),
        "rmse": float(np.sqrt(np.square(y - yhat).mean())),
        "wape_pct": float(100 * delta.sum() / wape_den) if wape_den else None,
        "smape_pct": float(100 * np.mean(np.divide(2 * delta, sm_den,
                                                  out=np.zeros_like(delta), where=sm_den > 1e-12))),
        "r2": float(1 - np.square(y - yhat).sum() / ss_total) if ss_total else None,
    }


def load_pharma_splits(
    source: pd.DataFrame,
    category: str,
    horizon_days: int,
    target_col: str = "quantity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra e valida treino/teste inteiramente em memória.

    Aceita o caminho do Parquet ou um DataFrame já carregado, por exemplo
    via ParquetRepository.load(). Lê apenas as colunas necessárias e não
    cria qualquer arquivo intermediário ou CSV.
    """
    if horizon_days not in ALLOWED_HORIZONS:
        raise ValueError(f"horizon_days deve ser um de {ALLOWED_HORIZONS}.")
    required = {"category", "horizon_days", "split", "date", target_col, "forecast_origin"}
        
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no Parquet/DataFrame: {sorted(missing)}")
    frame = source.loc[
        (source["category"] == category) & (source["horizon_days"] == horizon_days),
        ["date", "category", "horizon_days", "forecast_origin", "split", target_col],
    ].copy()
    if frame.empty:
        raise ValueError(f"Não há registros para category={category!r}, horizon_days={horizon_days}.")
    unexpected = set(frame["split"].dropna().unique()) - {"train", "test"}
    if unexpected or frame["split"].isna().any():
        raise ValueError(f"Valores inesperados de split: {unexpected}")

    # esta um pouco redundate mas antes isso que deixar passar
    train = frame[(frame["category"] == category) & (frame["horizon_days"] == horizon_days) & (frame["split"] == "train")]
    test =  frame[(frame["category"] == category) & (frame["horizon_days"] == horizon_days) & (frame["split"] == "test")]
    
    last_train = train["date"].iloc[-1]
    first_test = test["date"].iloc[0]

    if first_test != last_train + pd.Timedelta(days=1):
        raise ValueError(f"Split não contíguo/cronológico: fim treino={last_train.date()}, "
                         f"início teste={first_test.date()}.")
    return train, test


def _as_series(dfx: pd.DataFrame, target_col: str) -> pd.Series:
    s = dfx.set_index("date")[target_col].astype(float).asfreq("D")
    s.name = target_col
    return s


def _baseline_prediction(history: pd.Series, horizon: int) -> float:
    if len(history) < 7:
        raise ValueError("Baseline sazonal exige >=7 observações.")
    return float(history.iloc[-7:].to_numpy()[((horizon - 1) % 7)])


class SARIMAXRegressor:
    """Modelo estatístico SARIMAX persistível; sem tuning via conjunto de teste.

    ``fit(train_df)``: ajusta parâmetros apenas no split train.
    ``evaluate(test_df)``: avaliação D+h por forecast_origin.
    ``evaluate_blocks(test_df)``: previsão em blocos até h dias (outra métrica).
    ``predict(horizon)``: previsão a partir da última data vista no treino.
    """
    def __init__(self, category: str = "N02BE", horizon_days: int = 15,
                 target_col: str = "quantity", order: tuple[int, int, int] = ORDER,
                 seasonal_order: tuple[int, int, int, int] = SEASONAL_ORDER,
                 trend: str = TREND, maxiter: int = 200):
        if horizon_days not in ALLOWED_HORIZONS:
            raise ValueError(f"horizon_days deve estar em {ALLOWED_HORIZONS}")
        self.category = category
        self.horizon_days = int(horizon_days)
        self.target_col = target_col
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.maxiter = int(maxiter)
        self.result: Optional[SARIMAXResults] = None
        self._train: Optional[pd.Series] = None
        self.last_train_date: Optional[pd.Timestamp] = None

    def fit_model(self, train_df: pd.DataFrame) -> "SARIMAXRegressor":
        
        # if "split" in train_df.columns and not train_df["split"].eq("train").all():
        #     raise ValueError("fit() recebeu linhas fora do split='train'.")
        
        if "category" in train_df.columns and not train_df["category"].eq(self.category).all():
            raise ValueError("O DataFrame contém categoria diferente da configurada.")
        
        # if "horizon_days" in train_df.columns and not train_df["horizon_days"].eq(self.horizon_days).all():
        #     raise ValueError("O DataFrame contém horizonte diferente do configurado.")
        
        train = _as_series(train_df, self.target_col)
        if len(train) < 365:
            raise ValueError("O treino precisa de pelo menos 365 observações diárias.")
        
        model = SARIMAX(
            endog=train,
            exog=calendar_exog(train.index),
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.result = model.fit(disp=False, maxiter=self.maxiter)
        self._train = train
        self.last_train_date = train.index[-1]
        if not self.converged:
            warnings.warn("SARIMAX não convergiu. Revise maxiter e os parâmetros antes do uso.", RuntimeWarning)
        return self

    @property
    def converged(self) -> bool:
        return bool(self.result is not None and self.result.mle_retvals.get("converged", False))

    def _require_fit(self):
        if self.result is None or self._train is None or self.last_train_date is None:
            raise RuntimeError("Treine com fit() ou carregue um modelo salvo com load() antes.")

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        """Previsão realizada na última data do treino, para os próximos h dias."""
        self._require_fit()
        h = self.horizon_days if horizon is None else int(horizon)
        if h < 1:
            raise ValueError("horizon precisa ser >= 1")
        dates = pd.date_range(self.last_train_date + pd.Timedelta(days=1), periods=h, freq="D")
        p = self.result.get_forecast(steps=h, exog=calendar_exog(dates))
        ci = p.conf_int(alpha=0.05).to_numpy()
        return pd.DataFrame({
            "forecast_origin": self.last_train_date,
            "date": dates,
            "step_ahead": np.arange(1, h + 1),
            "horizon_days": h,
            "prediction": np.maximum(p.predicted_mean.to_numpy(), 0),
            "ci95_lower": np.maximum(ci[:, 0], 0),
            "ci95_upper": np.maximum(ci[:, 1], 0),
        })

    def _verify_test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        self._require_fit()
        # f = _validate_frame(test_df, self.target_col)
        if "split" in test_df.columns and not test_df["split"].eq("test").all():
            raise ValueError("evaluate() aceita somente o split=test.")
        
        if "category" in test_df.columns and not test_df["category"].eq(self.category).all():
            raise ValueError("Categoria de teste difere do treino.")
        
        if "horizon_days" in test_df.columns and not test_df["horizon_days"].eq(self.horizon_days).all():
            raise ValueError("Horizonte de teste difere do treino.")
        
        if test_df["date"].iloc[0] != self.last_train_date + pd.Timedelta(days=1):
            raise ValueError("O teste precisa começar no dia seguinte ao fim do treino.")
        
        return test_df

    def evaluate(self, test_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """Mede a previsão exatamente D+h. Atualiza estado apenas ao chegar à origem.

        Os primeiros h-1 alvos de teste NÃO podem ser previstos honestamente por
        um modelo ajustado até o último dia de treino: suas origens precedem esse
        treinamento. Esses registros são excluídos, não usados como validação.
        Os parâmetros SARIMAX permanecem fixos: ``extend`` só atualiza o filtro.
        """
        df = self._verify_test(test_df)
        h = self.horizon_days
        scored = df.loc[df["date"] - pd.Timedelta(days=h) >= self.last_train_date].copy()
        if scored.empty:
            raise ValueError("Nenhuma data de teste é elegível ao horizonte fixo.")
        if "forecast_origin" in df.columns:
            if not scored["forecast_origin"].equals(
                (scored["date"] - pd.Timedelta(days=h)).rename("forecast_origin")
            ):
                raise ValueError("forecast_origin inválida no teste.")
        observations = df.set_index("date")[self.target_col].astype(float)
        state = self.result
        state_end = self.last_train_date
        rows = []
        for r in scored.itertuples(index=False):
            date = r.date
            origin = date - pd.Timedelta(days=h)
            if origin > state_end:
                new_dates = pd.date_range(state_end + pd.Timedelta(days=1), origin, freq="D")
                revealed = observations.reindex(new_dates)
                if revealed.isna().any():
                    raise ValueError("Há observações indisponíveis no instante de atualização.")
                revealed.index.name = "date"
                state = state.extend(endog=revealed, exog=calendar_exog(new_dates))
                state_end = origin
            future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")
            forecast = state.get_forecast(steps=h, exog=calendar_exog(future_dates))
            # previsão em D+h, e NÃO média dos passos 1..h
            prediction = max(0., float(forecast.predicted_mean.iloc[-1]))
            ci = forecast.conf_int(alpha=.05).iloc[-1]
            hist = pd.concat([self._train, observations.loc[:origin]]).sort_index()
            rows.append({
                "forecast_origin": origin, "date": date,
                "horizon_days": h, "step_ahead": h,
                "actual": float(getattr(r, self.target_col)),
                "prediction": prediction,
                "baseline_seasonal_naive": _baseline_prediction(hist, h),
                "ci95_lower": max(0., float(ci.iloc[0])),
                "ci95_upper": max(0., float(ci.iloc[1])),
            })
        preds = pd.DataFrame(rows)
        metrics = {
            "evaluation": "exact_horizon_D_plus_h",
            "horizon_days": h,
            "test_rows": len(df),
            "scored_rows": len(preds),
            "excluded_first_test_rows": len(df) - len(preds),
            "excluded_reason": "forecast_origin anterior ao fim do treino",
            "sarimax": quality_metrics(preds["actual"], preds["prediction"]),
            "seasonal_naive": quality_metrics(preds["actual"], preds["baseline_seasonal_naive"]),
        }
        return metrics, preds

    def evaluate_blocks(self, test_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """Backtest em blocos consecutivos (previsões D+1,...,D+h por bloco).

        Útil para verificar todos os dias do teste; NÃO compara diretamente com
        ML supervisionado cuja linha representa sempre D+h.
        """
        df = self._verify_test(test_df)
        test = _as_series(df, self.target_col)
        h = self.horizon_days
        state = self.result
        rows = []
        for offset in range(0, len(test), h):
            obs = test.iloc[offset:offset+h]
            origin = self.last_train_date if offset == 0 else test.index[offset - 1]
            forecast = state.get_forecast(steps=len(obs), exog=calendar_exog(obs.index))
            history = pd.concat([self._train, test.iloc[:offset]])
            pred = np.maximum(forecast.predicted_mean.to_numpy(), 0.)
            for step, (day, actual, estimate) in enumerate(zip(obs.index, obs.to_numpy(), pred), 1):
                rows.append({
                    "forecast_origin": origin, "date": day,
                    "horizon_days": h, "step_ahead": step,
                    "actual": float(actual), "prediction": float(estimate),
                    "baseline_seasonal_naive": _baseline_prediction(history, step),
                })
            state = state.extend(endog=obs, exog=calendar_exog(obs.index))
        preds = pd.DataFrame(rows)
        metrics = {
            "evaluation": "rolling_nonoverlapping_blocks_steps_1_to_h",
            "horizon_days": h,
            "test_rows": len(df),
            "scored_rows": len(preds),
            "sarimax": quality_metrics(preds["actual"], preds["prediction"]),
            "seasonal_naive": quality_metrics(preds["actual"], preds["baseline_seasonal_naive"]),
        }
        return metrics, preds

    def save(self, output_dir: str | Path) -> Path:
        """Salva o modelo efetivamente ajustado e o manifesto para auditoria."""
        self._require_fit()
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "sarimax_treino.pkl"
        self.result.save(str(path))
        metadata = {
            "category": self.category, "horizon_days": self.horizon_days,
            "target_col": self.target_col, "order": list(self.order),
            "seasonal_order": list(self.seasonal_order), "trend": self.trend,
            "maxiter": self.maxiter,
            "exogenous_columns": list(calendar_exog(self._train.index).columns),
            "train_start": str(self._train.index.min().date()),
            "train_end": str(self._train.index.max().date()),
            "train_rows": len(self._train),
            "converged": self.converged,
            "aic": float(self.result.aic),
            "model_file": path.name,
            "model_scope": "somente split train (modelo de avaliacao)",
        }
        (folder / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load_model(cls, output_dir: str | Path) -> "SARIMAXRegressor":
        folder = Path(output_dir)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self = cls(category=metadata["category"], horizon_days=metadata["horizon_days"],
                   target_col=metadata["target_col"], order=tuple(metadata["order"]),
                   seasonal_order=tuple(metadata["seasonal_order"]), trend=metadata["trend"],
                   maxiter=metadata.get("maxiter", 200))
        self.result = SARIMAXResults.load(str(folder / metadata["model_file"]))
        train = self.result.model.data.orig_endog
        if isinstance(train, pd.DataFrame):
            train = train.iloc[:, 0]
        if not isinstance(train, pd.Series):
            raise TypeError("Arquivo salvo não preservou a série temporal original do treino.")
        self._train = train.asfreq("D").astype(float)
        self.last_train_date = self._train.index.max()
        if str(self.last_train_date.date()) != metadata["train_end"]:
            raise ValueError("Incompatibilidade entre datas do modelo e manifesto.")
        return self

    def evaluate(self, model:SARIMAXRegressor,
            parquet_or_df: str | Path | pd.DataFrame,
            *,
            category: str = "N02BE",
            horizon_days: int = 15,
            target_col: str = "quantity",
            maxiter: int = 200,
        ) -> tuple[SARIMAXRegressor, dict, dict[str, pd.DataFrame]]:
            """Pipeline inteiramente em memória (não persiste modelos nem previsões).
    
            Retorna (modelo, métricas, dataframes), onde ``dataframes`` contém
            'exact_horizon', 'block_backtest' e 'forecast_after_train'.
            Chame model.save(caminho) explicitamente se quiser persistir o modelo.
            """
            train, test = load_pharma_splits(parquet_or_df, category, horizon_days, target_col)
            model = SARIMAXRegressor(
                category=category, horizon_days=horizon_days,
                target_col=target_col, maxiter=maxiter,
            ).fit_model(train)
            
            metrics_exact, preds_exact = model.evaluate(test)
            metrics_blocks, preds_blocks = model.evaluate_blocks(test)
            metrics = {
                "category": category,
                "horizon_days": horizon_days,
                "converged": model.converged,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_start": str(train["date"].min().date()),
                "train_end": str(train["date"].max().date()),
                "test_start": str(test["date"].min().date()),
                "test_end": str(test["date"].max().date()),
                "exact_horizon": metrics_exact,
                "block_backtest": metrics_blocks,
            }
            predictions = {
                "exact_horizon": preds_exact,
                "block_backtest": preds_blocks,
                "forecast_after_train": model.predict(),
            }
            return model, metrics, predictions

    def train_and_evaluate(self,
        parquet_or_df: str | Path | pd.DataFrame,
        *,
        category: str = "N02BE",
        horizon_days: int = 15,
        target_col: str = "quantity",
        maxiter: int = 200,
    ) -> tuple[SARIMAXRegressor, dict, dict[str, pd.DataFrame]]:
        """Pipeline inteiramente em memória (não persiste modelos nem previsões).

        Retorna (modelo, métricas, dataframes), onde ``dataframes`` contém
        'exact_horizon', 'block_backtest' e 'forecast_after_train'.
        Chame model.save(caminho) explicitamente se quiser persistir o modelo.
        """
        train, test = load_pharma_splits(parquet_or_df, category, horizon_days, target_col)
        model = SARIMAXRegressor(
            category=category, horizon_days=horizon_days,
            target_col=target_col, maxiter=maxiter,
        ).fit_model(train)
        metrics_exact, preds_exact = model.evaluate(test)
        metrics_blocks, preds_blocks = model.evaluate_blocks(test)
        metrics = {
            "category": category,
            "horizon_days": horizon_days,
            "converged": model.converged,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": str(train["date"].min().date()),
            "train_end": str(train["date"].max().date()),
            "test_start": str(test["date"].min().date()),
            "test_end": str(test["date"].max().date()),
            "exact_horizon": metrics_exact,
            "block_backtest": metrics_blocks,
        }
        predictions = {
            "exact_horizon": preds_exact,
            "block_backtest": preds_blocks,
            "forecast_after_train": model.predict(),
        }
        return model, metrics, predictions


    def save_predictions_as_parquet(predictions: dict[str, pd.DataFrame], output_path: str | Path) -> Path:
        """Opcional: combina os resultados em UM Parquet, nunca em CSV."""
        frames = []
        for evaluation, df in predictions.items():
            item = df.copy()
            item.insert(0, "evaluation", evaluation)
            frames.append(item)
        combined = pd.concat(frames, ignore_index=True, sort=False)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            combined.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError("Instale pyarrow para gravar Parquet: pip install pyarrow") from exc
        return path


    def tune_hyperparameters(
            self,
            df_train: pd.DataFrame,
            orders: Optional[list[tuple[int, int, int]]] = None,
            seasonal_orders: Optional[list[tuple[int, int, int, int]]] = None,
            n_splits: int = 2,
            validation_days: int = 45,
        ) -> tuple["SARIMAXRegressor", dict]:
            """Busca o menor RMSE D+h em janelas temporais, sem usar o teste final.
    
            Prioriza modelos que convergiram em todos os folds. Se nenhum convergir,
            admite o melhor ajuste com previsões válidas, mas emite um alerta.
            """
            from itertools import product
    
            orders = orders or [(1, 0, 1), (1, 1, 1), (2, 0, 1), (2, 1, 1)]
            seasonal_orders = seasonal_orders or [
                (0, 0, 0, 7), (1, 0, 0, 7),
                (0, 1, 1, 7), (1, 1, 0, 7),
            ]
            #data = df_train.sort_values("date").reset_index(drop=True)

            data = (
                df_train.loc[
                    (df_train["category"] == self.category)
                    & (df_train["horizon_days"] == self.horizon_days)
                ]
                .sort_values("date")
                .reset_index(drop=True)
            )

            if validation_days <= self.horizon_days or len(data) < 365 + n_splits * validation_days:
                raise ValueError("Janela temporal insuficiente para o tuning D+h.")
    
            self.cv_results_ = []
            self.cv_failures_ = []
    
            for order, seasonal_order in product(orders, seasonal_orders):
                fold_rmse = []
                folds_converged = True
                for fold in range(n_splits):
                    end = len(data) - (n_splits - fold) * validation_days
                    fold_train = data.iloc[:end]
                    fold_valid = data.iloc[end:end + validation_days].copy()
                    fold_valid["split"] = "test"
                    candidate = SARIMAXRegressor(
                        category=self.category,
                        horizon_days=self.horizon_days,
                        target_col=self.target_col,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend=self.trend,
                        maxiter=self.maxiter,
                    )
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            candidate.fit_model(fold_train)
                        folds_converged &= candidate.converged
                        metrics, _ = candidate.evaluate(fold_valid)
                        rmse = float(metrics["sarimax"]["rmse"])
                        if not np.isfinite(rmse):
                            raise ValueError("RMSE inválido (NaN/inf)")
                        fold_rmse.append(rmse)
                    except Exception as exc:
                        self.cv_failures_.append({
                            "order": order,
                            "seasonal_order": seasonal_order,
                            "fold": fold + 1,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        break
    
                if len(fold_rmse) == n_splits:
                    self.cv_results_.append({
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "rmse": float(np.sqrt(np.mean(np.square(fold_rmse)))),
                        "converged": bool(folds_converged),
                    })
    
            if not self.cv_results_:
                details = "\n".join(str(x) for x in self.cv_failures_[:3])
                raise RuntimeError(
                    "Nenhuma configuração gerou previsões válidas em todos os folds. "
                    f"Primeiros erros:\n{details}"
                )
    
            # Convergência vem antes do RMSE; ajustes não convergidos são fallback.
            best = min(self.cv_results_, key=lambda r: (not r["converged"], r["rmse"]))
            self.order = best["order"]
            self.seasonal_order = best["seasonal_order"]
            self.best_cv_rmse = best["rmse"]
            self.best_cv_converged = best["converged"]
            self.best_params = {
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "trend": self.trend,
                "maxiter": self.maxiter,
            }
            if not self.best_cv_converged:
                warnings.warn(
                    "Nenhuma configuração convergiu em todos os folds. "
                    "Foi selecionado o menor RMSE entre os ajustes com previsões válidas, "
                    "mas o resultado precisa de revisão (ex.: aumentar maxiter).",
                    RuntimeWarning,
                )
    
            self.fit_model(data)  # Reajusta somente sobre todo o split de treino.
            return self, self.best_params.copy()

    def _params_path(self, path: str | Path | None) -> Path:
        folder = Path(path) if path is not None else (
            Path(f"{MODELS_PARAM}\\sarimax_best_params.json")
        )
        return folder if folder.suffix == ".json" else folder / "sarimax_best_params.json"

    def save_best_params(self) -> Path:
        """Persiste apenas a configuração selecionada e sua métrica de validação."""
        destination = self._params_path(f"{MODELS_PARAM}\\sarimax_best_params.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "category": self.category,
            "horizon_days": self.horizon_days,
            "best_params": self.best_params,
            "best_cv_rmse": self.best_cv_rmse,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def load_params(self) -> dict:
        """Carrega parâmetros; executar fit_model() para treinar com eles."""
        local = f"{MODELS_PARAM}\\sarimax_best_params.json"
        local = local.replace("\\","/")
        payload = json.loads(self._params_path(local).read_text(encoding="utf-8"))

        if (payload["category"]) != (self.category):
            raise ValueError("Parâmetros pertencem a outra categoria ou horizonte.")

        params = payload["best_params"]
        self.order = tuple(params["order"])
        self.seasonal_order = tuple(params["seasonal_order"])
        self.trend = params["trend"]
        self.maxiter = params["maxiter"]
        self.best_params = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "maxiter": self.maxiter,
        }
        self.best_cv_rmse = payload.get("best_cv_rmse")
        self.result = self._train = self.last_train_date = None
        return self.best_params.copy()