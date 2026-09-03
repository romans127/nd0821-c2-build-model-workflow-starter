#!/usr/bin/env python
"""
Download a raw data artifact from W&B, apply basic cleaning, and upload the
result as a new artifact.

Cleaning steps:
- drop listings with price outside [min_price, max_price]
- drop listings outside the NYC bounding box
- convert last_review to datetime
- drop rows missing last_review, reviews_per_month, neighbourhood_group or
  room_type
"""
import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

KEY_COLUMNS = ["last_review", "reviews_per_month", "neighbourhood_group", "room_type"]


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f"Fetching input artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logger.info(f"Loaded {len(df)} rows")

    # Drop listings priced outside the accepted band
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info(
        f"Dropped {(~idx).sum()} rows with price outside "
        f"[{args.min_price}, {args.max_price}]"
    )

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    logger.info(f"Dropped {(~idx).sum()} rows outside NYC boundaries")

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Drop rows missing key columns
    before = len(df)
    df = df.dropna(subset=KEY_COLUMNS)
    logger.info(f"Dropped {before - len(df)} rows with missing {KEY_COLUMNS}")

    logger.info(f"Saving {len(df)} cleaned rows to clean_data.csv")
    df.to_csv("clean_data.csv", index=False)

    logger.info(f"Uploading {args.output_artifact} to Weights & Biases")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_data.csv")
    run.log_artifact(artifact)

    os.remove("clean_data.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of the raw data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact (raw data) in W&B",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact (cleaned data)",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A brief description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum accepted price; listings below this are dropped",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum accepted price; listings above this are dropped",
        required=True,
    )

    args = parser.parse_args()

    go(args)
