from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "RainTomorrow"
DATE_COLUMN = "Date"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the source date into calendar features and remove the raw date."""
    features = frame.drop(columns=[TARGET_COLUMN], errors="ignore").copy()

    if DATE_COLUMN in features.columns:
        raw_dates = features.pop(DATE_COLUMN)
        if pd.api.types.is_numeric_dtype(raw_dates):
            # Excel stores dates as days from this base date.
            dates = pd.to_datetime(
                raw_dates,
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )
        else:
            dates = pd.to_datetime(raw_dates, errors="coerce")

        features["Year"] = dates.dt.year
        features["Month"] = dates.dt.month
        features["Day"] = dates.dt.day

    return features
