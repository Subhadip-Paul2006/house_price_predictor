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

# SEED fix kar rahe taaki random numbers har baar same hi milein
SEED = 42
N_SAMPLES = 1000
OUTPUT_PATH = os.path.join("data", "house_data.csv")

# Location ke hisab se rates adjust karne ke liye multiplier set kiya hai (Downtown sabse mehanga, Rural sasta)
LOCATION_MULTIPLIERS = {
    "Downtown": 2.0,
    "Urban": 1.6,
    "Suburban": 1.2,
    "Rural": 0.8,
}


def generate_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame with synthetic house-price data."""

    # RNG setup kar rahe hain seed ke sath
    rng = np.random.default_rng(seed)

    # --- Feature values randomly generate kar rahe hain ---
    area = rng.integers(500, 5000, size=n)                    # Square feet (area)
    bedrooms = rng.integers(1, 7, size=n)                     # 1 se 6 BHK tak
    bathrooms = rng.integers(1, 6, size=n)                     # 1 se 5 bathrooms
    age = rng.integers(0, 51, size=n)                         # Ghar kitna purana hai (0-50 years)
    locations = rng.choice(
        list(LOCATION_MULTIPLIERS.keys()),
        size=n,
        p=[0.15, 0.35, 0.35, 0.15],                           # Kaunsi location kitni baar aani chahiye
    )

    # --- Ek basic price formula set kar rahe hain (Lakhs me) ---
    # Area, bedrooms, bathrooms ke hisab se price badhega, age ke hisab se depreciate hoga (kam hoga)
    base_price_per_sqft = 0.035          
    price = (
        area * base_price_per_sqft
        + bedrooms * 5.0                  # Har ek bedroom ka +5 Lakh
        + bathrooms * 3.0                 # Har ek bathroom ka +3 Lakh
        - age * 0.3                       # Har saal 0.3 Lakh value kam hogi (purana hone ki wajah se)
    )

    # Location multiplier apply kar rahe hain
    location_mult = [LOCATION_MULTIPLIERS[loc] for loc in locations]
    price = price * location_mult

    # Thoda real-world market noise add kar rahe hain (around 10% fluctuation)
    noise = rng.normal(1.0, 0.10, size=n)
    price = price * noise

    # Kuch bhi ho jaye, price 5 Lakh se kam nahi hona chahiye (free me toh koi ghar nahi dega)
    price = np.maximum(price, 5.0)

    # Decimal values ko 2 places tak round kar rahe hain
    price = np.round(price, 2)

    df = pd.DataFrame({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": locations,
        "Price": price,
    })

    # --- Jaanबूझkar kuch missing/NaN values daal rahe hain taaki preprocessor clean karna seekhe ---
    # Area me 2%, Bedrooms me 1%, aur Location me 1.5% NaNs daal rahe hain
    n_area_nan = int(0.02 * n)
    n_beds_nan = int(0.01 * n)
    n_loc_nan = int(0.015 * n)

    df.loc[rng.choice(n, n_area_nan, replace=False), "Area"] = np.nan
    df.loc[rng.choice(n, n_beds_nan, replace=False), "Bedrooms"] = np.nan
    df.loc[rng.choice(n, n_loc_nan, replace=False), "Location"] = np.nan

    return df


def main() -> None:
    # Target directory exist nahi karti toh pehle folder create karo
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Generated {len(df)} rows → {OUTPUT_PATH}")
    print(f"   Missing values:\n{df.isnull().sum()}")
    print(f"\n   Sample rows:\n{df.head()}")
    print(f"\n   Price statistics:\n{df['Price'].describe()}")


if __name__ == "__main__":
    main()
