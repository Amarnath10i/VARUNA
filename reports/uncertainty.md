# Prediction intervals and the flood trigger

90% split-conformal intervals calibrated on 1981-1990 predictions and checked on 2016-2020. *On flood days*: days above the catchment's 99th percentile, the days a warning or an insurance trigger is about. Widths are relative to the catchment's mean flow.

## Regional LSTM simulation

| intervals                          | coverage   | on flood days   | below median flow   | catchments within 85-95%   |   median width / mean flow |   flood days: width / observed flow |
|:-----------------------------------|:-----------|:----------------|:--------------------|:---------------------------|---------------------------:|------------------------------------:|
| one width for all (current VARUNA) | 87.4%      | 8.2%            | 95.9%               | 35%                        |                       1.87 |                                0.07 |
| log scale, per catchment           | 73.5%      | 56.5%           | 62.4%               | 24%                        |                       1.37 |                                1.27 |
| log scale, per flow regime         | 78.6%      | 60.3%           | 71.4%               | 27%                        |                       1.72 |                                2.48 |
| log scale, per catchment, adaptive | 84.2%      | 64.0%           | 78.6%               | 54%                        |                       1.72 |                                1.66 |

### Trigger backtest, regional lstm simulation

| fires when                                        | intervals                          |   POD |   FAR |   CSI | onsets warned   |   triggers per catchment-year |
|:--------------------------------------------------|:-----------------------------------|------:|------:|------:|:----------------|------------------------------:|
| point forecast                                    | -                                  |  0.34 |  0.49 |  0.26 | 29% of 645      |                          0.44 |
| lower bound and confident (RiskEngine as written) | one width for all (current VARUNA) |  0.26 |  0.44 |  0.22 | 21% of 645      |                          0.31 |
| lower bound                                       | one width for all (current VARUNA) |  0.28 |  0.44 |  0.23 | 23% of 645      |                          0.33 |
| upper bound                                       | one width for all (current VARUNA) |  0.43 |  0.72 |  0.21 | 39% of 645      |                          1    |
| lower bound and confident (RiskEngine as written) | log scale, per flow regime         |  0    |  0    |  0    | 0% of 645       |                          0    |
| lower bound                                       | log scale, per flow regime         |  0.03 |  0.38 |  0.03 | 1% of 645       |                          0.03 |
| upper bound                                       | log scale, per flow regime         |  0.9  |  0.83 |  0.17 | 93% of 645      |                          3.32 |
| lower bound and confident (RiskEngine as written) | log scale, per catchment, adaptive |  0    |  0    |  0    | 0% of 645       |                          0    |
| lower bound                                       | log scale, per catchment, adaptive |  0.01 |  0.29 |  0.01 | 0% of 645       |                          0.01 |
| upper bound                                       | log scale, per catchment, adaptive |  0.9  |  0.84 |  0.16 | 92% of 645      |                          3.62 |
| P(flood) >= 0.10, cut-off tuned on 1981-1990      | log scale, per flow regime         |  0.63 |  0.66 |  0.29 | 62% of 645      |                          1.19 |

## Forecast 1 day ahead

| intervals                          | coverage   | on flood days   | below median flow   | catchments within 85-95%   |   median width / mean flow |   flood days: width / observed flow |
|:-----------------------------------|:-----------|:----------------|:--------------------|:---------------------------|---------------------------:|------------------------------------:|
| one width for all (current VARUNA) | 88.6%      | 6.5%            | 97.8%               | 35%                        |                       1.05 |                                0.04 |
| log scale, per catchment           | 81.9%      | 46.5%           | 81.2%               | 42%                        |                       0.19 |                                0.59 |
| log scale, per flow regime         | 83.5%      | 59.8%           | 79.6%               | 40%                        |                       0.19 |                                1.08 |
| log scale, per catchment, adaptive | 87.0%      | 61.7%           | 86.3%               | 73%                        |                       0.23 |                                1.04 |

### Trigger backtest, forecast 1 day ahead

| fires when                                        | intervals                          |   POD |   FAR |   CSI | onsets warned   |   triggers per catchment-year |
|:--------------------------------------------------|:-----------------------------------|------:|------:|------:|:----------------|------------------------------:|
| point forecast                                    | -                                  |  0.51 |  0.36 |  0.4  | 26% of 645      |                          0.55 |
| lower bound and confident (RiskEngine as written) | one width for all (current VARUNA) |  0.46 |  0.32 |  0.38 | 23% of 645      |                          0.47 |
| lower bound                                       | one width for all (current VARUNA) |  0.47 |  0.33 |  0.38 | 23% of 645      |                          0.48 |
| upper bound                                       | one width for all (current VARUNA) |  0.56 |  0.49 |  0.37 | 33% of 645      |                          0.73 |
| lower bound and confident (RiskEngine as written) | log scale, per flow regime         |  0    |  0    |  0    | 0% of 645       |                          0    |
| lower bound                                       | log scale, per flow regime         |  0.13 |  0.12 |  0.13 | 3% of 645       |                          0.11 |
| upper bound                                       | log scale, per flow regime         |  0.87 |  0.73 |  0.26 | 75% of 645      |                          2.09 |
| lower bound and confident (RiskEngine as written) | log scale, per catchment, adaptive |  0    |  0    |  0    | 0% of 645       |                          0    |
| lower bound                                       | log scale, per catchment, adaptive |  0.13 |  0.2  |  0.13 | 4% of 645       |                          0.12 |
| upper bound                                       | log scale, per catchment, adaptive |  0.85 |  0.73 |  0.26 | 72% of 645      |                          2.05 |
| P(flood) >= 0.15, cut-off tuned on 1981-1990      | log scale, per flow regime         |  0.66 |  0.46 |  0.42 | 43% of 645      |                          0.81 |

## Forecast 3 day ahead

| intervals                          | coverage   | on flood days   | below median flow   | catchments within 85-95%   |   median width / mean flow |   flood days: width / observed flow |
|:-----------------------------------|:-----------|:----------------|:--------------------|:---------------------------|---------------------------:|------------------------------------:|
| one width for all (current VARUNA) | 88.5%      | 6.5%            | 97.7%               | 34%                        |                       1.42 |                                0.05 |
| log scale, per catchment           | 81.0%      | 45.3%           | 78.6%               | 37%                        |                       0.33 |                                0.67 |
| log scale, per flow regime         | 82.6%      | 56.6%           | 77.8%               | 38%                        |                       0.36 |                                1.15 |
| log scale, per catchment, adaptive | 86.2%      | 58.2%           | 84.2%               | 62%                        |                       0.41 |                                1.1  |

### Trigger backtest, forecast 3 day ahead

| fires when                                        | intervals                          |   POD |   FAR |   CSI | onsets warned   |   triggers per catchment-year |
|:--------------------------------------------------|:-----------------------------------|------:|------:|------:|:----------------|------------------------------:|
| point forecast                                    | -                                  |  0.41 |  0.41 |  0.32 | 29% of 645      |                          0.48 |
| lower bound and confident (RiskEngine as written) | one width for all (current VARUNA) |  0.33 |  0.38 |  0.28 | 24% of 645      |                          0.38 |
| lower bound                                       | one width for all (current VARUNA) |  0.35 |  0.38 |  0.29 | 26% of 645      |                          0.4  |
| upper bound                                       | one width for all (current VARUNA) |  0.49 |  0.6  |  0.28 | 38% of 645      |                          0.83 |
| lower bound and confident (RiskEngine as written) | log scale, per flow regime         |  0    |  0    |  0    | 0% of 645       |                          0    |
| lower bound                                       | log scale, per flow regime         |  0.04 |  0.19 |  0.04 | 2% of 645       |                          0.04 |
| upper bound                                       | log scale, per flow regime         |  0.87 |  0.8  |  0.2  | 85% of 645      |                          2.79 |
| lower bound and confident (RiskEngine as written) | log scale, per catchment, adaptive |  0    |  0    |  0    | 0% of 645       |                          0    |
| lower bound                                       | log scale, per catchment, adaptive |  0.06 |  0.17 |  0.06 | 3% of 645       |                          0.04 |
| upper bound                                       | log scale, per catchment, adaptive |  0.83 |  0.79 |  0.2  | 80% of 645      |                          2.66 |
| P(flood) >= 0.15, cut-off tuned on 1981-1990      | log scale, per flow regime         |  0.6  |  0.52 |  0.36 | 49% of 645      |                          0.86 |
