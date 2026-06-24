"""
generate_data.py — One-time script to create a synthetic house-price dataset.

Generates ~1000 rows of Indian real-estate data with realistic feature
distributions and a few intentional NaN values so that the preprocessing
pipeline has something to clean.

Usage:
    python generate_data.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 42
N_SAMPLES = 1000
OUTPUT_PATH = os.path.join("data", "house_data.csv")

# Location multipliers (Downtown is most expensive, Rural cheapest)
LOCATION_MULTIPLIERS = {
    "Downtown": 2.0,
    "Urban": 1.6,
    "Suburban": 1.2,
    "Rural": 0.8,
}


def generate_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame with synthetic house-price data."""

    rng = np.random.default_rng(seed)

    # --- Feature distributions ---
    area = rng.integers(500, 5000, size=n)                    # sq ft
    bedrooms = rng.integers(1, 7, size=n)                     # 1–6
    bathrooms = rng.integers(1, 6, size=n)                     # 1–5
    age = rng.integers(0, 51, size=n)                         # 0–50 years
    locations = rng.choice(
        list(LOCATION_MULTIPLIERS.keys()),
        size=n,
        p=[0.15, 0.35, 0.35, 0.15],                           # distribution
    )

    # --- Price formula (Lakhs ₹) ---
    # Base price from area, with contributions from beds/baths/age
    base_price_per_sqft = 0.035          # Lakhs per sq ft base rate
    price = (
        area * base_price_per_sqft
        + bedrooms * 5.0                  # each bedroom adds ~5 L
        + bathrooms * 3.0                 # each bathroom adds ~3 L
        - age * 0.3                       # depreciation per year
    )

    # Apply location multiplier
    location_mult = [LOCATION_MULTIPLIERS[loc] for loc in locations]
    price = price * location_mult

    # Add realistic noise (~10 %)
    noise = rng.normal(1.0, 0.10, size=n)
    price = price * noise

    # Floor prices at 5 Lakh (no free houses)
    price = np.maximum(price, 5.0)

    # Round to 2 decimal places
    price = np.round(price, 2)

    df = pd.DataFrame({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": locations,
        "Price": price,
    })

    # --- Inject some missing values for preprocessing testing ---
    # ~2 % of values in Area, ~1 % in Bedrooms, ~1.5 % in Location
    n_area_nan = int(0.02 * n)
    n_beds_nan = int(0.01 * n)
    n_loc_nan = int(0.015 * n)

    df.loc[rng.choice(n, n_area_nan, replace=False), "Area"] = np.nan
    df.loc[rng.choice(n, n_beds_nan, replace=False), "Bedrooms"] = np.nan
    df.loc[rng.choice(n, n_loc_nan, replace=False), "Location"] = np.nan

    return df


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Generated {len(df)} rows → {OUTPUT_PATH}")
    print(f"   Missing values:\n{df.isnull().sum()}")
    print(f"\n   Sample rows:\n{df.head()}")
    print(f"\n   Price statistics:\n{df['Price'].describe()}")


if __name__ == "__main__":
    main()
