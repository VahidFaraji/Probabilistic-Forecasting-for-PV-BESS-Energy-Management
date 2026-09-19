import pandas as pd


STEPS_PER_DAY = 48
VALIDATION_DAYS = 30
TEST_DAYS = 30

DEFAULT_VALIDATION_SIZE = VALIDATION_DAYS * STEPS_PER_DAY
DEFAULT_TEST_SIZE = TEST_DAYS * STEPS_PER_DAY


def split_by_time(
    data: pd.DataFrame,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically using common timestamp boundaries."""
    if "ds" not in data.columns:
        raise ValueError("The data must contain a 'ds' column.")

    split_data = data.copy()
    split_data["ds"] = pd.to_datetime(split_data["ds"])

    sort_columns = ["unique_id", "ds"] if "unique_id" in split_data.columns else ["ds"]
    split_data = split_data.sort_values(sort_columns).reset_index(drop=True)

    timestamps = split_data["ds"].drop_duplicates().sort_values().reset_index(drop=True)

    if validation_size + test_size >= len(timestamps):
        raise ValueError("Validation and test periods leave no data for training.")

    validation_start = timestamps.iloc[-(validation_size + test_size)]
    test_start = timestamps.iloc[-test_size]

    train_data = split_data.loc[split_data["ds"] < validation_start].copy()
    validation_data = split_data.loc[
        (split_data["ds"] >= validation_start)
        & (split_data["ds"] < test_start)
    ].copy()
    test_data = split_data.loc[split_data["ds"] >= test_start].copy()

    return train_data, validation_data, test_data
