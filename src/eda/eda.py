"""Quick EDA on the raw NYC Airbnb sample to inform the basic_cleaning step.

Checks:
- price distribution and outliers
- latitude/longitude spread (flags points outside the NYC bounding box)
- missing values per column
- dtype of last_review
"""

import os

import pandas as pd

SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", "get_data", "data", "sample1.csv"
)

# Reasonable bounding box for NYC
LON_MIN, LON_MAX = -74.25, -73.50
LAT_MIN, LAT_MAX = 40.5, 41.2


def main():
    df = pd.read_csv(SAMPLE_PATH)
    print(f"Shape: {df.shape}\n")

    print("=== dtypes ===")
    print(df.dtypes)
    print(f"\nlast_review dtype: {df['last_review'].dtype}")

    print("\n=== price distribution ===")
    print(df["price"].describe())
    for q in (0.95, 0.99, 0.999):
        print(f"price p{q * 100:g}: {df['price'].quantile(q):.2f}")
    print(f"rows with price == 0: {(df['price'] == 0).sum()}")
    print(f"rows with price < 10: {(df['price'] < 10).sum()}")
    print(f"rows with price > 350: {(df['price'] > 350).sum()}")

    print("\n=== lat/long spread ===")
    print(df[["latitude", "longitude"]].describe())
    outside = df[
        ~df["longitude"].between(LON_MIN, LON_MAX)
        | ~df["latitude"].between(LAT_MIN, LAT_MAX)
    ]
    print(
        f"rows outside lon [{LON_MIN}, {LON_MAX}] / lat [{LAT_MIN}, {LAT_MAX}]: "
        f"{len(outside)}"
    )
    if len(outside):
        print(outside[["name", "neighbourhood_group", "latitude", "longitude"]].head(10))

    print("\n=== missing values per column ===")
    missing = df.isna().sum()
    print(missing[missing > 0])


if __name__ == "__main__":
    main()
