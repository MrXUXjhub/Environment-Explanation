# Hourly input table

Place your local modelling table at `data/GE.csv`, or edit `DATA_FILE` in a model script.

Required columns:

| Role | Columns |
| --- | --- |
| Time | `timestamp`, or `date` together with `Hour of Day` |
| Predictors | `Day of Week`, `Hour of Day`, `O3`, `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `Temperature`, `Wind Speed`, `Pressure`, `Humidity`, `Precipitation` |
| Targets | `SGE`, `EGE`, `GiniStart`, `GiniEnd` |

Each row represents one unique hour. Predictors and targets must be numeric and finite. `Hour of Day` uses integers 0–23 and must agree with the timestamp. Weekday codes retain the source convention.

Edit `TARGETS` at the top of a script if modelling a subset of outcomes. Model outputs are written under `results/`.
