# Streamflow simulation, 2016-2020

Daily flow from forcing alone (no observed flow as input), scored on the test years 2016-2020, which no model was fitted or selected on. One row per model; medians over catchments with at least a year of observed flow in the test years; intervals from 2,000 bootstrap resamples of catchments. Catchments 3076, 4050, 15023 are excluded: their recorded runoff exceeds 1.5 times their rainfall (see the last table).

## All catchments

| model                                            |   catchments | median NSE (95% CI)    |   median KGE | NSE > 0   | peak-flow bias (FHV)   |
|:-------------------------------------------------|-------------:|:-----------------------|-------------:|:----------|:-----------------------|
| Climatology (day-of-year mean flow)              |          202 | 0.072 (0.030 to 0.119) |        0.077 | 63%       | -73%                   |
| GR4J, calibrated per catchment                   |          202 | 0.319 (0.146 to 0.413) |        0.363 | 71%       | -35%                   |
| CAMELS-IND regional LSTM (published predictions) |          202 | 0.416 (0.347 to 0.489) |        0.445 | 77%       | -32%                   |
| LightGBM on engineered rainfall memory           |          202 | 0.430 (0.338 to 0.505) |        0.399 | 79%       | -30%                   |
| Regional LSTM without catchment attributes       |          202 | 0.323 (0.189 to 0.433) |        0.301 | 70%       | -35%                   |
| Regional LSTM, one seed                          |          202 | 0.397 (0.325 to 0.467) |        0.423 | 71%       | -25%                   |
| **Regional LSTM, 3-seed ensemble**               |          202 | 0.441 (0.346 to 0.511) |        0.409 | 75%       | -26%                   |

## Catchments covered by the 47-model conceptual benchmark

| model                                              |   catchments | median NSE (95% CI)    |   median KGE | NSE > 0   | peak-flow bias (FHV)   |
|:---------------------------------------------------|-------------:|:-----------------------|-------------:|:----------|:-----------------------|
| Climatology (day-of-year mean flow)                |          141 | 0.110 (0.048 to 0.168) |        0.128 | 67%       | -73%                   |
| GR4J, calibrated per catchment                     |          141 | 0.388 (0.253 to 0.466) |        0.407 | 77%       | -36%                   |
| Best of 47 conceptual models, chosen per catchment |          141 | 0.359 (0.196 to 0.425) |        0.389 | 73%       | -35%                   |
| CAMELS-IND regional LSTM (published predictions)   |          141 | 0.489 (0.397 to 0.544) |        0.481 | 84%       | -33%                   |
| LightGBM on engineered rainfall memory             |          141 | 0.462 (0.394 to 0.537) |        0.443 | 85%       | -34%                   |
| Regional LSTM without catchment attributes         |          141 | 0.397 (0.265 to 0.508) |        0.395 | 77%       | -36%                   |
| Regional LSTM, one seed                            |          141 | 0.456 (0.355 to 0.517) |        0.506 | 77%       | -27%                   |
| **Regional LSTM, 3-seed ensemble**                 |          141 | 0.495 (0.375 to 0.558) |        0.47  | 81%       | -28%                   |

## Regional LSTM, 3-seed ensemble against each model, catchment by catchment

| versus                                             |   catchments | median NSE gain (95% CI)   | catchments improved   |
|:---------------------------------------------------|-------------:|:---------------------------|:----------------------|
| Climatology (day-of-year mean flow)                |          202 | +0.267 (+0.202 to +0.320)  | 81%                   |
| GR4J, calibrated per catchment                     |          202 | +0.040 (+0.013 to +0.078)  | 63%                   |
| Best of 47 conceptual models, chosen per catchment |          141 | +0.042 (+0.022 to +0.074)  | 67%                   |
| CAMELS-IND regional LSTM (published predictions)   |          202 | +0.000 (-0.013 to +0.010)  | 50%                   |
| LightGBM on engineered rainfall memory             |          202 | -0.006 (-0.019 to +0.007)  | 47%                   |
| Regional LSTM without catchment attributes         |          202 | +0.048 (+0.020 to +0.075)  | 63%                   |
| Regional LSTM, one seed                            |          202 | +0.011 (+0.007 to +0.021)  | 67%                   |

## By regulation and climate: Regional LSTM, 3-seed ensemble, paired with each model

| split      | group                               |   catchments |   median NSE | gain over GR4J, calibrated per catchment   | gain over LightGBM on engineered rainfall memory   | gain over CAMELS-IND regional LSTM (published predictions)   |
|:-----------|:------------------------------------|-------------:|-------------:|:-------------------------------------------|:---------------------------------------------------|:-------------------------------------------------------------|
| regulation | reservoir index < 0.1               |           45 |        0.472 | +0.018 (+0.00 to +0.06)                    | +0.006 (-0.02 to +0.01)                            | +0.002 (-0.04 to +0.01)                                      |
| regulation | 0.1 to 1                            |           90 |        0.544 | +0.057 (+0.01 to +0.10)                    | -0.004 (-0.02 to +0.01)                            | +0.004 (-0.01 to +0.03)                                      |
| regulation | > 1 (storage exceeds a year's flow) |           45 |        0.068 | +0.097 (+0.01 to +0.19)                    | -0.016 (-0.15 to +0.02)                            | -0.011 (-0.18 to +0.04)                                      |
| regulation | unknown                             |           22 |        0.501 | -0.017 (-0.11 to +0.10)                    | -0.028 (-0.12 to +0.03)                            | -0.000 (-0.06 to +0.08)                                      |
| aridity    | dry (P/PET <= 0.60)                 |           56 |        0.035 | -0.016 (-0.08 to +0.08)                    | -0.090 (-0.27 to -0.01)                            | -0.118 (-0.19 to +0.03)                                      |
| aridity    | middle                              |           73 |        0.453 | +0.040 (+0.01 to +0.12)                    | -0.010 (-0.05 to +0.02)                            | -0.003 (-0.05 to +0.01)                                      |
| aridity    | wet (P/PET > 0.87)                  |           73 |        0.591 | +0.054 (+0.01 to +0.10)                    | +0.007 (-0.01 to +0.01)                            | +0.012 (-0.00 to +0.03)                                      |

## Ungauged catchments

Each catchment predicted by models that never saw its flows: 5 folds of whole river basins (Godavari alone is one fold), so no test gauge shares a river network with a training gauge. Random 5-fold over gauges, the usual protocol, is shown for comparison: nested gauges can then sit on both sides.

| model                                                           |   catchments | median NSE (95% CI)    |   median KGE | NSE > 0   | peak-flow bias (FHV)   |
|:----------------------------------------------------------------|-------------:|:-----------------------|-------------:|:----------|:-----------------------|
| GR4J calibrated on the catchment's own flows (gauged reference) |          202 | 0.319 (0.146 to 0.413) |        0.363 | 71%       | -35%                   |
| Regional LSTM trained with the catchment (gauged reference)     |          202 | 0.441 (0.346 to 0.511) |        0.409 | 75%       | -26%                   |
| GR4J with parameters of the 5 most similar catchments           |          202 | 0.178 (0.055 to 0.302) |        0.098 | 62%       | -34%                   |
| LightGBM, catchment's river basin held out                      |          202 | 0.317 (0.135 to 0.386) |        0.206 | 65%       | -33%                   |
| **Regional LSTM, catchment's river basin held out**             |          202 | 0.284 (0.145 to 0.393) |        0.285 | 66%       | -32%                   |
| GR4J donors, random folds                                       |          202 | 0.244 (0.060 to 0.348) |        0.209 | 64%       | -33%                   |
| LightGBM, random folds                                          |          202 | 0.294 (0.181 to 0.396) |        0.31  | 69%       | -31%                   |
| Regional LSTM, random folds                                     |          202 | 0.358 (0.257 to 0.431) |        0.335 | 71%       | -29%                   |

| versus                                                          |   catchments | median NSE gain (95% CI)   | catchments improved   |
|:----------------------------------------------------------------|-------------:|:---------------------------|:----------------------|
| GR4J calibrated on the catchment's own flows (gauged reference) |          202 | -0.012 (-0.040 to +0.005)  | 45%                   |
| Regional LSTM trained with the catchment (gauged reference)     |          202 | -0.057 (-0.085 to -0.030)  | 29%                   |
| GR4J with parameters of the 5 most similar catchments           |          202 | +0.042 (+0.015 to +0.075)  | 63%                   |
| LightGBM, catchment's river basin held out                      |          202 | +0.016 (-0.006 to +0.029)  | 53%                   |
| GR4J donors, random folds                                       |          202 | +0.008 (-0.005 to +0.035)  | 55%                   |
| LightGBM, random folds                                          |          202 | -0.013 (-0.050 to +0.001)  | 44%                   |
| Regional LSTM, random folds                                     |          202 | -0.021 (-0.047 to -0.003)  | 41%                   |

## Excluded: runoff above 1.5 times rainfall (NSE)

|   gauge |   runoff / rainfall |   Climatology (day-of-year mean flow) |   GR4J, calibrated per catchment |   LightGBM on engineered rainfall memory |   Regional LSTM, 3-seed ensemble |   Regional LSTM without catchment attributes |   CAMELS-IND regional LSTM (published predictions) |   Regional LSTM, one seed |   Best of 47 conceptual models, chosen per catchment |
|--------:|--------------------:|--------------------------------------:|---------------------------------:|-----------------------------------------:|---------------------------------:|---------------------------------------------:|---------------------------------------------------:|--------------------------:|-----------------------------------------------------:|
|    3076 |                1.53 |                                 0.18  |                           -0.097 |                                   -0.146 |                           -0.264 |                                       -0.241 |                                             -0.391 |                    -0.35  |                                              nan     |
|    4050 |                2.55 |                                -0.151 |                           -0.145 |                                   -0.144 |                           -0.136 |                                       -0.136 |                                             -0.137 |                    -0.131 |                                               -0.147 |
|   15023 |                1.78 |                                 0.486 |                            0.397 |                                    0.442 |                            0.613 |                                        0.527 |                                              0.738 |                     0.633 |                                                0.611 |
