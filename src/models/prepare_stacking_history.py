"""Historico diario para o stacking PharmaDemand (versao tolerante a divergencias).

O dataframe supervisionado permanece INTACTO. Quando D+7 e D+15 discordam
na mesma data-alvo, nao se soma nem se calcula media: para uma demonstracao
exploratoria, um horizonte e usado como fonte provisoria do historico.

RECOMENDADO: passe history_df com o historico diario ORIGINAL de vendas,
com colunas ['date','category','quantity']. So esse dado evita a escolha
arbitraria de um horizonte. As divergencias dos alvos sao auditadas.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

KEYS = ['date', 'category', 'horizon_days']


def target_conflicts(df: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """Registros nos quais quantidades de horizontes diferentes discordam na MESMA data-alvo."""
    data = df.loc[df['horizon_days'].isin([7, 15]), KEYS + ['quantity']].copy()
    data['date'] = pd.to_datetime(data['date']).dt.normalize()
    data['target_date'] = data['date'] + pd.to_timedelta(data['horizon_days'], unit='D')
    agg = data.groupby(['category', 'target_date'], as_index=False).agg(
        min_quantity=('quantity', 'min'), max_quantity=('quantity', 'max'),
        num_horizons=('horizon_days', 'nunique'),
    )
    agg['difference'] = agg['max_quantity'] - agg['min_quantity']
    return agg.loc[
        (agg['num_horizons'] > 1) & (agg['difference'] > tolerance)
    ].sort_values('difference', ascending=False).reset_index(drop=True)


def build_daily_history(
    df: pd.DataFrame,
    fill_method: str = 'ffill',
    tolerance: float = 1e-6,
    *,
    history_df: pd.DataFrame | None = None,
    prefer_horizon: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (daily, gaps_audit) sem modificar D+7/D+15 do `df`.

    history_df: fonte diaria externa recomendada (date/category/quantity).
    Sem ela, prefere D+7 ou D+15 em conflitos (APENAS exploratorio).
    A serie auxiliar nao corrige divergencias dos targets do ensemble.
    """
    required = set(KEYS + ['quantity', 'split'])
    if missing := sorted(required - set(df.columns)):
        raise ValueError(f'Faltam colunas no df: {missing}')
    if fill_method not in ('ffill', 'zero', 'interpolate'):
        raise ValueError("fill_method deve ser 'ffill', 'zero' ou 'interpolate'.")
    if prefer_horizon not in (7, 15):
        raise ValueError('prefer_horizon deve ser 7 ou 15.')

    data = df.loc[
        df['split'].isin(['train', 'test']) & df['horizon_days'].isin([7, 15])
    ].copy()
    data['date'] = pd.to_datetime(data['date'], errors='raise').dt.normalize()
    data['category'] = data['category'].astype(str)
    data['quantity'] = pd.to_numeric(data['quantity'], errors='raise')
    data['horizon_days'] = data['horizon_days'].astype(int)
    if data.empty or data[KEYS + ['quantity']].isna().any().any():
        raise ValueError('O dataframe supervisionado esta vazio ou contem chaves/targets nulos.')
    if data.duplicated(KEYS).any():
        raise ValueError('Chaves date/category/horizon_days duplicadas no dataframe supervisionado.')

    conflicts = target_conflicts(data, tolerance=tolerance)
    if len(conflicts):
        warnings.warn(
            f'{len(conflicts)} datas-alvo possuem alvos D+7/D+15 divergentes. '
            'A divergencia foi REGISTRADA, nao corrigida; verifique a semantica '
            'dos alvos antes de interpretar metricas do ensemble.\n'
            + conflicts.head(3).to_string(index=False),
            RuntimeWarning, stacklevel=2,
        )

    if history_df is not None:
        # Fonte independente: preferivel para SARIMAX e AutoTheta.
        missing_history = sorted({'date', 'category', 'quantity'} - set(history_df.columns))
        if missing_history:
            raise ValueError(f'Faltam colunas no historico: {missing_history}')
        observed = history_df[['date', 'category', 'quantity']].copy()
        observed['date'] = pd.to_datetime(observed['date'], errors='raise').dt.normalize()
        observed['category'] = observed['category'].astype(str)
        observed['quantity'] = pd.to_numeric(observed['quantity'], errors='raise')
        if observed.duplicated(['date', 'category']).any():
            raise ValueError('history_df contem datas duplicadas por categoria; agregue antes.')
        origin = 'historico_diario_original'
    else:
        # Fallback EXPLORATORIO: desambiguar SEM SOMAR horizontes.
        # Usa date+h somente quando o contrato supervisionado realmente e t+h.
        proxy = data[KEYS + ['quantity']].copy()
        proxy['date'] = proxy['date'] + pd.to_timedelta(proxy['horizon_days'], unit='D')
        proxy['_priority'] = proxy['horizon_days'].ne(prefer_horizon).astype(int)
        observed = (proxy.sort_values(['category', 'date', '_priority'])
                    .drop_duplicates(['category', 'date'], keep='first')
                    [['date', 'category', 'quantity']].copy())
        origin = f'proxy_D{prefer_horizon}_nas_datas_alvo'
        warnings.warn(
            f'Usando historico PROVISORIO derivado de D+{prefer_horizon}. '
            'Isso contorna a excecao, mas nao comprova que quantity e venda diaria. '
            'Para resultados finais, passe history_df com as vendas diarias reais.',
            RuntimeWarning, stacklevel=2,
        )

    parts, audits = [], []
    for cat, rows in observed.groupby('category', sort=True):
        series = (rows.set_index('date')['quantity']
                  .sort_index().asfreq('D'))
        mask = series.isna()
        if fill_method == 'ffill':
            filled = series.ffill()
        elif fill_method == 'zero':
            filled = series.fillna(0.0)
        else:
            warnings.warn('Interpolacao utiliza valores posteriores e pode causar leakage.',
                          RuntimeWarning, stacklevel=2)
            filled = series.interpolate(method='time', limit_area='inside')
        if filled.isna().any():
            raise ValueError(f'{cat}: sobraram {int(filled.isna().sum())} dias sem quantity.')
        parts.append(pd.DataFrame({
            'date': series.index, 'category': cat,
            'quantity': filled.to_numpy(dtype=float),
            'was_imputed': mask.to_numpy(dtype='int8'),
        }))
        audits.append({
            'category': cat, 'days': len(series),
            'missing_days_filled': int(mask.sum()),
            'first_day': series.index.min(), 'last_day': series.index.max(),
            'method': fill_method, 'source': origin,
            'target_conflicts_total': len(conflicts),
        })
    if not parts:
        raise ValueError('Nao ha nenhum historico diario para construir.')
    daily = pd.concat(parts, ignore_index=True).sort_values(
        ['category', 'date']).reset_index(drop=True)
    return daily, pd.DataFrame(audits)
