# Local input data

Place the hourly modelling table at `data/GE.csv`, or pass its location to
`prepare_data.py --data /path/to/GE.csv`.

The observations are not included in this code update. `data/GE.csv`, trained
models and generated results are ignored by Git. The original providers and
access routes are listed in the repository README.

Required columns:

| Role | Columns |
| --- | --- |
| Time | `timestamp`, or `date` together with `Hour of Day` |
| Predictors | `Day of Week`, `Hour of Day`, `O3`, `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `Temperature`, `Wind Speed`, `Pressure`, `Humidity`, `Precipitation` |
| Outcomes | `SGE`, `EGE`, `GiniStart`, `GiniEnd` |

Each row represents one unique hour. Predictors and outcomes must be numeric and
finite. `Hour of Day` uses integers 0–23. A `timestamp` must agree with this hour
column. Weekday codes retain the source convention. Columns such as `ordercount`,
`Trip Volume`, `Ln_Trip Volume` and `Travel Distance` may remain in the local CSV;
the model matrix is constructed only from the explicit 13-predictor whitelist.

Rows are sorted by time, duplicate hours are rejected and missing hours are not
imputed. A target is eligible only when its current and preceding 23 input hours
are continuous. This common eligibility rule also applies to the table models
so that all six models are compared at identical target times.

For the exact author-confirmed CSV fingerprint in `config.py`, a 2023 date-entry
error is corrected to 2017 in derived timestamps only. Other files retain their
years unless a justified `--year` is explicitly supplied. The source CSV is
never overwritten. Retain original measurement units; the scripts do not infer
or convert units.

`self_check.py` works without this file: its main checks use clearly labelled,
generated test fixtures. An additional author-data regression check runs only
when the exact local author CSV is available.
