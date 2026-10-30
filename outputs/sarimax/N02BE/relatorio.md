# PharmaDemand — Avaliação SARIMAX (N02BE)

- Dados: `salesdaily.csv`, série diária.
- Treino: **02/01/2014 a 08/04/2019** (1923 dias).
- Teste: **09/04/2019 a 08/10/2019** (183 dias).
- Modelo: SARIMAX(1, 1, 1) x (1, 0, 0, 7); regressoras de calendário (fim de semana e seno/cosseno anual).
- Convergência: **True**.

| Experimento | Modelo | MAE | RMSE | WAPE | sMAPE |
|---|---|---:|---:|---:|---:|
| holdout_6_meses | sarimax | 8.09 | 11.07 | 35.30% | 37.34% |
| holdout_6_meses | seasonal_naive | 11.30 | 14.91 | 49.27% | 49.30% |
| backtest_7_dias | sarimax | 7.12 | 9.58 | 31.03% | 32.25% |
| backtest_7_dias | seasonal_naive | 9.31 | 12.46 | 40.58% | 42.60% |
| backtest_15_dias | sarimax | 7.34 | 9.77 | 32.02% | 33.07% |
| backtest_15_dias | seasonal_naive | 9.41 | 12.86 | 41.02% | 42.21% |

## Interpretação

O **holdout** prevê os seis meses de uma vez, sem atualizar o modelo com observações do teste.
Nos **backtests de 7 e 15 dias**, a cada origem o modelo prevê uma janela e, depois,
recebe apenas as vendas já observadas para atualizar o estado estatístico **sem refit**.
A baseline *seasonal naive* repete os últimos sete valores disponíveis.

**Não use este conjunto de teste para selecionar hiperparâmetros:** se desejar otimizar
ordens e regressoras, crie uma validação temporal anterior ao bloco final de teste.
Os valores são mantidos como no dataset, com decimais, sem arredondamento de unidades.

O artefato `sarimax_treino.pkl` usa somente o treino. Para implantação após
validação, recomenda-se reestimar um modelo final com toda a série disponível,
guardando separadamente as métricas de teste aqui apresentadas.
