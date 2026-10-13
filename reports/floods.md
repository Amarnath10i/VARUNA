# Flood events, 2016-2020

A flood day is a day above the catchment's 99th percentile of observed 1991-2015 flow (about 3.7 days a year); an event is a run of flood days. An observed event is detected if the model is above the threshold on any of its days; a model event that overlaps no observed event is a false alarm. *Onsets warned*: the model was above the threshold on the first day of the event, i.e. for a forecast, before the river got there. Counts are pooled over catchments; intervals resample catchments.

## Simulation from weather alone

| model                                            |   events | POD (95% CI)     | FAR (95% CI)     |   CSI | onsets warned   |
|:-------------------------------------------------|---------:|:-----------------|:-----------------|------:|:----------------|
| Climatology (day-of-year mean flow)              |      803 | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) |  0    | 0% of 645       |
| GR4J, calibrated per catchment                   |      803 | 0.28 (0.23-0.33) | 0.53 (0.47-0.58) |  0.21 | 26% of 645      |
| CAMELS-IND regional LSTM (published predictions) |      803 | 0.33 (0.28-0.37) | 0.48 (0.42-0.53) |  0.25 | 28% of 645      |
| LightGBM on engineered rainfall memory           |      803 | 0.32 (0.27-0.37) | 0.50 (0.45-0.55) |  0.24 | 30% of 645      |
| Regional LSTM, 3-seed ensemble                   |      803 | 0.34 (0.29-0.39) | 0.49 (0.44-0.55) |  0.26 | 29% of 645      |

## Forecasts that also know today's flow

Rain between the issue day and the target day is the observed rain (a perfect rainfall forecast), so these are upper bounds.

| model                                    |   events | POD (95% CI)     | FAR (95% CI)     |   CSI | onsets warned   |
|:-----------------------------------------|---------:|:-----------------|:-----------------|------:|:----------------|
| Persistence, 1 day ahead                 |      803 | 0.50 (0.44-0.55) | 0.47 (0.42-0.53) |  0.34 | 0% of 645       |
| LightGBM with observed flow, 1 day ahead |      803 | 0.51 (0.46-0.56) | 0.36 (0.31-0.42) |  0.4  | 26% of 645      |
| Persistence, 2 day ahead                 |      803 | 0.40 (0.34-0.45) | 0.60 (0.54-0.66) |  0.25 | 11% of 645      |
| LightGBM with observed flow, 2 day ahead |      803 | 0.44 (0.39-0.49) | 0.38 (0.32-0.43) |  0.35 | 31% of 645      |
| Persistence, 3 day ahead                 |      803 | 0.32 (0.26-0.38) | 0.67 (0.61-0.74) |  0.19 | 11% of 645      |
| LightGBM with observed flow, 3 day ahead |      803 | 0.41 (0.36-0.45) | 0.41 (0.35-0.47) |  0.32 | 29% of 645      |
