"""Gera 8 Parquets OOF/holdout para o notebook 06_ridge_stacking.ipynb.

Requer um dataset supervisionado em que:
  date = ORIGEM da previsao; horizon_days in (7, 15)
  quantity = venda REAL em date + horizon_days
  split in ('train', 'test'); category = codigo ATC

IMPORTANTE: os hiperparametros de base sao fixados previamente; nao use uma
selecao feita olhando todos os folds de OOF (isso contaminaria o meta-treino).

SARIMAX e AutoTheta abaixo sao implementacoes de referencia. Para reproduzir
metricas de classes pessoais ja ajustadas, troque os factories/parametros,
respeitando o mesmo protocolo de backtesting temporal.

Requer: pip install pandas numpy scikit-learn lightgbm xgboost statsmodels statsforecast pyarrow
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------
# CONFIGURACAO (ajuste somente este bloco)
# -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # se o arquivo estiver em src/models/
FEATURES_FILE = ROOT / 'data/features/anomaly_predictions.parquet'
OUTPUT_DIR = ROOT / 'data/stacking'
HORIZONS = (7, 15)
N_SPLITS = 3
# Selecionar todas as origens = 1; usar 7 para prototipagem rapida SEM mudar
# as chaves entre os quatro modelos. Para o resultado final, mantenha 1.
ORIGIN_STEP = 1
MIN_HISTORY = 60
CLIP_NONNEGATIVE = True

# None = reconstruir o historico diario a partir dos labels D+7 e D+15.
# Somente e valido se quantity for de fato o alvo date + horizonte.
# Alternativamente, aponte para um parquet com date/category/quantity diarios.
DAILY_HISTORY_FILE = None
# Desabilitar somente se os alvos supervisionados diferem do historico diario.
# Nesse caso, as metricas do stacking NAO devem ser reportadas como validas
# ate revisar o contrato de quantity e date/horizon_days.
ENFORCE_TARGET_ALIGNMENT = True

NUMERIC_FEATURES = [
    'qty_lag_origin', 'qty_lag_origin_7',
    'qty_roll_mean_7', 'qty_roll_std_7',
    'qty_roll_mean_28', 'qty_roll_std_28',
    'qty_mean_ratio_7_28', 'qty_last_zscore_28',
    'qty_zero_ratio_28', 'weekday', 'annual_sin', 'annual_cos',
]
LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    max_depth=8, min_child_samples=20, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=3, n_jobs=1, verbosity=-1,
)
XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, objective='reg:squarederror',
    tree_method='hist', random_state=3, n_jobs=1,
)
SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 0, 0, 7)
THETA_SEASON_LENGTH = 7

KEYS = ['date', 'category', 'horizon_days']
MODELS = ['lightgbm', 'xgboost', 'sarimax', 'autotheta']


def _ensure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    needed = set(KEYS + ['quantity', 'split'])
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f'Dataset sem colunas: {missing}')
    data = frame.loc[frame['horizon_days'].isin(HORIZONS)].copy()
    data['date'] = pd.to_datetime(data['date'], errors='raise').dt.normalize()
    data['category'] = data['category'].astype('str')
    data['horizon_days'] = data['horizon_days'].astype(int)
    data['quantity'] = pd.to_numeric(data['quantity'], errors='raise')
    if data['quantity'].isna().any():
        raise ValueError('quantity contem nulos: verifique alinhamento do alvo.')
    if data['date'].isna().any():
        raise ValueError('date contem nulos.')
    if data.duplicated(KEYS).any():
        raise ValueError('Chaves duplicadas (date/category/horizon_days).')
    if not set(data['split']).issubset({'train', 'test'}):
        raise ValueError('split precisa usar train/test, como no notebook 05.')
    if set(data['split']) != {'train', 'test'}:
        raise ValueError('Nao encontrei train e test.')
    train = data.loc[data['split'].eq('train'), 'date']
    test = data.loc[data['split'].eq('test'), 'date']
    if train.max() >= test.min():
        raise ValueError('Holdout nao e estritamente posterior ao treino por origem.')
    for h in HORIZONS:
        for part in ('train', 'test'):
            if not data.loc[(data['horizon_days'] == h) & (data['split'] == part)].shape[0]:
                raise ValueError(f'D+{h}: split {part} vazio.')
    return data.sort_values(KEYS).reset_index(drop=True)


def relabel_origin_day_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Opcional: converter quantity(t) em quantity(t+h) usando a serie diaria.

    Execute SOMENTE quando quantity do parquet original for a venda do proprio
    dia `date`. Se ja representa t+h, nao chame esta funcao.
    A conversao pode usar observacoes de teste para definir o *alvo de avaliacao*,
    mas o treino por fold as purga para nunca aprender valores indisponiveis.
    """
    required = KEYS + ['quantity', 'split']
    if missing := sorted(set(required) - set(data.columns)):
        raise ValueError(f'Colunas ausentes: {missing}')
    data = data.copy()
    data['date'] = pd.to_datetime(data['date']).dt.normalize()
    quantities = data.groupby(['category', 'date'])['quantity'].agg(
        lambda x: float(x.max() - x.min())
    )
    if (quantities > 1e-6).any():
        raise ValueError('Os valores de quantity do proprio dia divergem entre horizontes; '
                         'nao e seguro derivar o observado diario desse DataFrame.')
    actual = data.groupby(['category', 'date'], as_index=False)['quantity'].first()
    actual = actual.rename(columns={'date': '_target_date', 'quantity': '_future_actual'})
    data['_target_date'] = data['date'] + pd.to_timedelta(data['horizon_days'], unit='D')
    data = data.merge(actual, on=['category', '_target_date'], how='left',
                      validate='many_to_one', sort=False)
    unlabelled = int(data['_future_actual'].isna().sum())
    if unlabelled:
        print(f'AVISO: descartadas {unlabelled} linhas cujo alvo futuro nao esta na serie.')
    data = data.dropna(subset=['_future_actual']).copy()
    data['quantity'] = data['_future_actual'].astype(float)
    return data.drop(columns=['_target_date', '_future_actual'])


def _daily_history(data: pd.DataFrame) -> pd.DataFrame:
    """Historico completo observado; nunca usar valores posteriores a origin."""
    if DAILY_HISTORY_FILE is not None:
        daily = pd.read_parquet(DAILY_HISTORY_FILE)[['date', 'category', 'quantity']].copy()
        daily['date'] = pd.to_datetime(daily['date']).dt.normalize()
        daily['category'] = daily['category'].astype(str)
        if daily.duplicated(['date', 'category']).any():
            raise ValueError('Historico diario com datas/categorias duplicadas.')
    else:
        # Construcao deterministica baseada no CONTRATO do target:
        # quantity na linha (origem=t, h) precisa ser o observado em t+h.
        daily = data[KEYS + ['quantity']].copy()
        daily['date'] += pd.to_timedelta(daily['horizon_days'], unit='D')
        disagreements = daily.groupby(['category', 'date'])['quantity'].agg(
            lambda x: float(x.max() - x.min())
        )
        if (disagreements > 1e-6).any():
            bad = disagreements[disagreements > 1e-6].head(5).to_dict()
            raise ValueError(
                'Targets dos horizontes nao coincidem na mesma data-alvo. '
                'POSSIVEL DESALINHAMENTO: quantity pode ser valor de date, '
                'e nao de date+h. Forneca DAILY_HISTORY_FILE e reconstrua '
                f'o target supervisionado corretamente. Exemplos: {bad}'
            )
        daily = daily.groupby(['category', 'date'], as_index=False)['quantity'].first()
    daily = daily.sort_values(['category', 'date']).reset_index(drop=True)
    if daily['quantity'].isna().any():
        raise ValueError('Historico diario com quantity nulo.')
    # Sem datas faltantes: SARIMAX/AutoTheta exigem passos diarios uniformes.
    for cat, sub in daily.groupby('category'):
        expected = pd.date_range(sub['date'].min(), sub['date'].max(), freq='D')
        if len(expected) != len(sub) or not sub['date'].reset_index(drop=True).equals(
            pd.Series(expected, name='date')
        ):
            raise ValueError(f'Serie {cat} tem lacunas. Preencha os dias no ETL '
                             'conforme regra de negocio (zero quando nao houve venda).')
    # Teste cruzado: quantity da linha supervisionada deve ser o observado em t+h.
    target = data[KEYS + ['quantity']].copy()
    target['target_date'] = target['date'] + pd.to_timedelta(target['horizon_days'], unit='D')
    matched = target.merge(
        daily.rename(columns={'date': 'target_date', 'quantity': 'observed'}),
        on=['category', 'target_date'], how='left', validate='many_to_one'
    )
    mismatch = (matched['observed'].isna() |
                ~np.isclose(matched['quantity'].to_numpy(dtype=float),
                            matched['observed'].to_numpy(dtype=float),
                            rtol=0, atol=1e-6, equal_nan=False))
    if mismatch.any():
        msg = (f'{int(mismatch.sum())} alvos supervisionados nao coincidem com '
               'o valor diario em date+h. Examine a definicao de quantity. '
               'Desabilitar ENFORCE_TARGET_ALIGNMENT apenas permite testes '
               'exploratorios, NAO valida as metricas OOF/holdout.')
        if ENFORCE_TARGET_ALIGNMENT:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return daily


def _selected_origins(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride < 1:
        raise ValueError('ORIGIN_STEP precisa ser >= 1.')
    dates = pd.DatetimeIndex(sorted(frame['date'].unique()))
    selected = dates[::stride]
    return frame[frame['date'].isin(selected)].copy()


def _folds(train: pd.DataFrame):
    dates = pd.DatetimeIndex(sorted(train['date'].unique()))
    for fold, (i_train, i_valid) in enumerate(
        TimeSeriesSplit(n_splits=N_SPLITS).split(dates), start=1
    ):
        valid_start = dates[i_valid[0]]
        valid_end = dates[i_valid[-1]]
        yield fold, valid_start, valid_end, train[
            train['date'].isin(dates[i_valid])
        ].copy()


def _create_tabular(name: str) -> Pipeline:
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor
        regressor = LGBMRegressor(**LGB_PARAMS)
    elif name == 'xgboost':
        from xgboost import XGBRegressor
        regressor = XGBRegressor(**XGB_PARAMS)
    else:
        raise ValueError(name)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', regressor),
    ])


def _safe_features(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(NUMERIC_FEATURES) - set(data.columns))
    if missing:
        raise ValueError('Features ausentes no dataset: ' + str(missing))
    X = data[NUMERIC_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    # NA aceita imputer somente nos valores disponiveis, sem imputacao global.
    return X


def _tabular_predict(name: str, data: pd.DataFrame, val: pd.DataFrame,
                     val_start: pd.Timestamp) -> np.ndarray:
    """Treino apenas com alvos publicados antes da primeira origem validada."""
    predictions = pd.Series(index=val.index, dtype=float)
    for h in HORIZONS:
        hold = val[val['horizon_days'] == h]
        if hold.empty:
            continue
        fit = data[
            (data['split'] == 'train') &
            (data['horizon_days'] == h) &
            (data['date'] + pd.Timedelta(days=h) < val_start)
        ]
        if fit.empty:
            raise ValueError(f'{name}: nao existem exemplos D+{h} antes de {val_start}.')
        estimator = _create_tabular(name)
        estimator.fit(_safe_features(fit), fit['quantity'].to_numpy())
        yhat = estimator.predict(_safe_features(hold))
        if CLIP_NONNEGATIVE:
            yhat = np.maximum(yhat, 0)
        predictions.loc[hold.index] = yhat
    if predictions.isna().any():
        raise RuntimeError(f'{name}: previsoes nulas em {val_start}.')
    return predictions.loc[val.index].to_numpy()


def _sparse_series_for_category(daily: pd.DataFrame, category: str) -> pd.Series:
    part = daily[daily['category'] == category]
    if part.empty:
        raise ValueError(f'Serie historica nao encontrada: {category}')
    return pd.Series(part['quantity'].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(part['date']), name='quantity')


def _time_series_predict(name: str, valid: pd.DataFrame, daily: pd.DataFrame,
                         anchor: pd.Timestamp) -> np.ndarray:
    """Forecast rolling-origin. Nenhum ponto > origin entra no historico.

    SARIMAX ajusta coeficientes antes da primeira origem e apenas atualiza
    os estados de filtragem com observacoes que ja ocorreram.
    AutoTheta reestima por origem (somente com observacoes ate a origem).
    """
    predictions = pd.Series(index=valid.index, dtype=float)
    max_h = max(HORIZONS)
    for category, group in valid.groupby('category', sort=False):
        history = _sparse_series_for_category(daily, category)
        origins = pd.DatetimeIndex(sorted(group['date'].unique()))
        if name == 'sarimax':
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            # Training strictly before first origin being predicted.
            init_history = history.loc[history.index < origins[0]]
            if len(init_history) < MIN_HISTORY:
                raise ValueError(f'SARIMAX {category}: historia insuficiente '
                                 f'({len(init_history)} < {MIN_HISTORY}).')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                state = SARIMAX(
                    init_history.to_numpy(), order=SARIMAX_ORDER,
                    seasonal_order=SARIMAX_SEASONAL_ORDER,
                    trend='c', enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=80)
            last_observed_date = init_history.index.max()
        elif name == 'autotheta':
            try:
                from statsforecast.models import AutoTheta
            except ImportError as exc:
                raise ImportError('Para usar AutoTheta: pip install statsforecast') from exc
        else:
            raise ValueError(name)

        for origin in origins:
            visible = history.loc[history.index <= origin]
            if len(visible) < MIN_HISTORY:
                raise ValueError(f'{name} {category}: historia insuficiente em {origin.date()}')
            if name == 'sarimax':
                new_data = history.loc[(history.index > last_observed_date)
                                       & (history.index <= origin)]
                if not new_data.empty:
                    state = state.extend(new_data.to_numpy())
                    last_observed_date = new_data.index.max()
                fc = np.asarray(state.forecast(steps=max_h), dtype=float)
            else:
                model = AutoTheta(season_length=THETA_SEASON_LENGTH)
                result = model.forecast(y=visible.to_numpy(dtype=np.float32), h=max_h)
                fc = np.asarray(result['mean'], dtype=float)
            if CLIP_NONNEGATIVE:
                fc = np.maximum(fc, 0)
            for h in HORIZONS:
                mask = (group['date'] == origin) & (group['horizon_days'] == h)
                if mask.any():
                    predictions.loc[group.index[mask]] = float(fc[h-1])
    if predictions.isna().any() or not np.isfinite(predictions.to_numpy()).all():
        raise RuntimeError(f'{name}: previsao ausente/nao finita: {valid.loc[predictions.isna(), KEYS].head()}')
    return predictions.loc[valid.index].to_numpy()


def _predict(name: str, train: pd.DataFrame, valid: pd.DataFrame,
             daily: pd.DataFrame, valid_start: pd.Timestamp) -> np.ndarray:
    if name in ('lightgbm', 'xgboost'):
        return _tabular_predict(name, train, valid, valid_start)
    return _time_series_predict(name, valid, daily, valid_start)


def _collect(data: pd.DataFrame, daily: pd.DataFrame):
    train = data.loc[data['split'] == 'train'].copy()
    holdout = data.loc[data['split'] == 'test'].copy()
    preds = {kind: {model: [] for model in MODELS} for kind in ('oof', 'holdout')}
    audit = []

    # OOF: MESMAS datas de validacao para todos os modelos/horizontes.
    for fold, start, end, valid_full in _folds(train):
        valid = _selected_origins(valid_full, ORIGIN_STEP)
        print(f'OOF fold {fold}: {start.date()} a {end.date()}, {len(valid)} linhas')
        for name in MODELS:
            pred = _predict(name, train, valid, daily, start)
            frame = valid[KEYS + ['quantity']].copy()
            frame['prediction'] = pred
            preds['oof'][name].append(frame)
            print(f'  {name}: {len(frame)} previsoes')
        for h in HORIZONS:
            fit = train[
                (train['horizon_days'] == h) &
                (train['date'] + pd.Timedelta(days=h) < start)
            ]
            audit.append({'kind': 'oof', 'fold': fold, 'horizon_days': h,
                          'last_available_target_date': (fit['date'] + pd.to_timedelta(h, unit='D')).max(),
                          'validation_start': start, 'validation_end': end,
                          'fit_rows_tabular': len(fit)})

    # Holdout: parametros de tabulares treinados APENAS com labels antes do
    # primeiro dia do holdout. Temporais: backtest rolling-origin usando
    # observacoes ja disponiveis em cada origem (sem conhecer futuro).
    test_start = holdout['date'].min()
    hold_selected = _selected_origins(holdout, ORIGIN_STEP)
    print(f'Holdout: {test_start.date()} a {holdout["date"].max().date()}, '
          f'{len(hold_selected)} linhas')
    for name in MODELS:
        frame = hold_selected[KEYS + ['quantity']].copy()
        frame['prediction'] = _predict(name, train, hold_selected, daily, test_start)
        preds['holdout'][name].append(frame)
        print(f'  {name}: {len(frame)} previsoes')
    for h in HORIZONS:
        fit = train[
            (train['horizon_days'] == h) &
            (train['date'] + pd.Timedelta(days=h) < test_start)
        ]
        audit.append({'kind': 'holdout', 'fold': 0, 'horizon_days': h,
                      'last_available_target_date': (fit['date'] + pd.to_timedelta(h, unit='D')).max(),
                      'validation_start': test_start, 'validation_end': holdout['date'].max(),
                      'fit_rows_tabular': len(fit)})
    return preds, pd.DataFrame(audit)


def generate(data: pd.DataFrame, *, persist=True):
    """Retorna dict[kind][name] DataFrames. persist=True salva os 8 Parquets."""
    data = _ensure_frame(data)
    daily = _daily_history(data)
    preds, audit = _collect(data, daily)
    outputs = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ('oof', 'holdout'):
        outputs[kind] = {}
        for name in MODELS:
            frame = pd.concat(preds[kind][name], ignore_index=True)
            frame = frame.sort_values(KEYS).reset_index(drop=True)
            if frame.duplicated(KEYS).any() or frame['prediction'].isna().any():
                raise RuntimeError(f'{kind}_{name}: chaves duplicadas ou predicoes nulas.')
            if not np.isfinite(frame['prediction'].to_numpy()).all():
                raise RuntimeError(f'{kind}_{name}: inf/NaN na previsao.')
            outputs[kind][name] = frame
        # Validar que todos os modelos tem a mesma chave e o mesmo alvo.
        ref = outputs[kind][MODELS[0]]
        for name in MODELS[1:]:
            other = outputs[kind][name]
            if not ref[KEYS].equals(other[KEYS]) or not np.allclose(
                ref['quantity'], other['quantity'], rtol=0, atol=1e-8
            ):
                raise AssertionError(f'{kind}_{name}: chaves/targets divergentes')
    if persist:
        # Persistir somente DEPOIS de todos os quatro modelos passarem nas
        # validacoes de chaves, alvos e finitude.
        for kind in ('oof', 'holdout'):
            for name in MODELS:
                path = OUTPUT_DIR / f'{kind}_{name}.parquet'
                outputs[kind][name].to_parquet(path, index=False)
                print('SALVO:', path, outputs[kind][name].shape)
        audit.to_csv(OUTPUT_DIR / 'generation_audit.csv', index=False)
        manifest = {
            'input': str(FEATURES_FILE),
            'target_semantics': 'quantity = observed sales at date + horizon_days',
            'model_names': MODELS, 'horizons': list(HORIZONS),
            'folds': N_SPLITS, 'origin_step': ORIGIN_STEP,
            'target_alignment_enforced': ENFORCE_TARGET_ALIGNMENT,
            'daily_history_file': str(DAILY_HISTORY_FILE),
            'sarimax_order': SARIMAX_ORDER,
            'sarimax_seasonal_order': SARIMAX_SEASONAL_ORDER,
            'autotheta_season_length': THETA_SEASON_LENGTH,
            'validation': 'expanding OOF; per-horizon label embargo; rolling-origin TS',
            'warning': 'Check that all raw features are past-only; no leakage audit of upstream feature ETL.',
        }
        (OUTPUT_DIR / 'generation_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
        )
    return outputs, audit


if __name__ == '__main__':
    print('Carregando:', FEATURES_FILE)
    result, audit_df = generate(pd.read_parquet(FEATURES_FILE), persist=True)
    print('\nAUDITORIA')
    print(audit_df.to_string(index=False))
    print('\n8 Parquets gerados em:', OUTPUT_DIR)
