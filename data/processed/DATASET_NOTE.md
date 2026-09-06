# Dataset note

The `features.csv` bundled with this repository is an **offline demonstration dataset** inherited from the original project and re-engineered with leakage-safe features.

For a real submission/deployment, run:

```bash
python -m src.feature_pipeline.pipeline --backfill --days 90
python -m src.training_pipeline.train
python -m src.training_pipeline.explainability
```

The backfill now retrieves real historical air-quality and weather data from Open-Meteo. It does not silently fall back to synthetic data. Synthetic data is available only through the explicit `--demo-backfill` option.
