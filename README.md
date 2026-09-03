# NYC Short-Term Rental Price Pipeline

An end-to-end machine learning pipeline that predicts nightly prices for short-term rental listings in New York City. I built this for WGU's D501 Machine Learning DevOps course. It downloads raw listing data, cleans and tests it, then splits it and trains a RandomForestRegressor, with every artifact and run tracked in Weights & Biases. MLflow orchestrates the steps and Hydra manages the configuration, so the whole thing retrains on new data with a single command.

## Links

- **GitHub repo:** https://github.com/romans127/nd0821-c2-build-model-workflow-starter
- **W&B project (public):** https://wandb.ai/rwatts7-western-governors-university/nyc_airbnb

The W&B project holds the full run history, artifact lineage, sweep results, and the feature importance plot for the production model.

## Pipeline overview

Each step is an MLflow component, run in this order:

1. **download** fetches the raw sample (`sample1.csv` or `sample2.csv`) and uploads it to W&B as a raw data artifact.
2. **basic_cleaning** drops price outliers outside $10 to $350, converts `last_review` to datetime, drops rows missing key columns, and (as of v1.0.1) drops listings outside the NYC bounding box of longitude -74.25 to -73.50 and latitude 40.5 to 41.2. It outputs `clean_data.csv`.
3. **data_check** is a pytest suite verifying row count (`test_row_count`), price range (`test_price_range`), geographic boundaries (`test_proper_boundaries`), and that the neighbourhood distribution matches a reference dataset within a KL divergence threshold (`test_similar_neigh_distrib`).
4. **data_split** splits the clean data into train/validation and test sets, stratified by `neighbourhood_group`.
5. **train_random_forest** trains the model and exports the full inference pipeline as the `random_forest_export` artifact.

A sixth step, `test_regression_model`, scores the production model against the held-out test set. It sits outside the default run because it needs a model export tagged `prod` before it can work.

Under the hood, the model is a RandomForestRegressor inside a sklearn Pipeline with a ColumnTransformer: OrdinalEncoder for `room_type`, SimpleImputer plus OneHotEncoder for `neighbourhood_group`, zero-imputation for the numeric features, a date-delta feature built from `last_review`, and TF-IDF on the listing `name`. The data itself has the fields you would expect from Airbnb-style listings: neighbourhood group, room type, location, review counts, and availability.

## Setup

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
wandb login
```

## How to run

Get your W&B API key from https://wandb.ai/authorize, then run the full pipeline from the repo root:

```bash
mlflow run .
```

You can also run a single step or a comma-separated subset, which is what I did most of the time while developing:

```bash
mlflow run . -P steps=basic_cleaning
mlflow run . -P steps=download,basic_cleaning
```

Config overrides go through Hydra. This is how I pointed the release at the second data sample:

```bash
mlflow run . -P hydra_options="etl.sample='sample2.csv'"
```

For the hyperparameter sweep I used Hydra's multirun mode, which trains one model per combination and logs each to W&B:

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
```

Testing the production model against the test set requires a `random_forest_export` artifact tagged `prod` in W&B, so it has to be run explicitly:

```bash
mlflow run . -P steps=test_regression_model
```

## Results

My sweep ran over `max_depth` and `n_estimators`. The best run was max_depth=50, n_estimators=200 with a validation MAE of 31.844, and I tagged that model export `prod` in W&B. Here is how the prod model scored on the held-out splits:

| Split | MAE | R2 |
|-------|-----|----|
| Validation | ~31.84 | ~0.569 |
| Test | ~32.25 | ~0.553 |

Test metrics land close to validation, so the model is not overfitting. One thing worth calling out: the TF-IDF features from the listing `name` column carry real weight. On the feature importance plot for the prod run, `name` sits at or near the top. The words hosts choose for their titles hold genuine pricing signal, and even a simple bag-of-words approach picks it up.

## Release history

**v1.0.0** was the first working release, trained on `sample1.csv` (~20k rows). Then I ran it against the new data sample. `sample2.csv` (~49k rows) failed at `data_check`: `test_proper_boundaries` caught one listing at latitude 40.4998, just outside the NYC bounding box. That was the tests doing exactly their job. New data brought a surprise, and the pipeline refused to train on it.

**v1.0.1** added a bounds filter to `basic_cleaning` dropping rows outside longitude -74.25 to -73.50 and latitude 40.5 to 41.2. Re-running on `sample2.csv` passed every check and trained a new model on the larger dataset.

That failure-to-fix loop is the point of the project. The data tests are not decoration; they caught a real data quality issue in production-style conditions and forced a versioned fix.

## Future improvements

- Richer EDA and visualizations. My current exploration is minimal, and price by neighbourhood or over time deserves a closer look.
- Models beyond RandomForest. Gradient boosting is the obvious next candidate.
- Better NLP on the `name` column. TF-IDF works, but bigrams or embeddings might pull more signal out of listing titles.
- Calibration. The model predicts a point estimate, and calibrated intervals would be more useful for actual pricing decisions.

## License

[License](LICENSE.txt)
