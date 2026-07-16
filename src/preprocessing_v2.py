import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed_v2"

WINDOW_SIZE = 30
RUL_CAP = 125
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_data(fd_id):
    """Load the raw NASA C-MAPSS files."""

    train_file = os.path.join(RAW_DIR, f"train_{fd_id}.txt")
    test_file = os.path.join(RAW_DIR, f"test_{fd_id}.txt")
    rul_file = os.path.join(RAW_DIR, f"RUL_{fd_id}.txt")

    train_df = pd.read_csv(train_file, sep=r"\s+", header=None)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None)

    column_names = (
        ["unit", "time"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    train_df.columns = column_names
    test_df.columns = column_names

    return train_df, test_df, rul_df


def add_training_rul(train_df):
    """Calculate Remaining Useful Life for every training cycle."""

    train_df = train_df.copy()

    maximum_cycles = (
        train_df.groupby("unit")["time"]
        .max()
        .rename("max_time")
    )

    train_df = train_df.join(maximum_cycles, on="unit")
    train_df["RUL"] = train_df["max_time"] - train_df["time"]

    # Piecewise-linear RUL target commonly used for C-MAPSS.
    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CAP)

    return train_df.drop(columns="max_time")


def split_training_units(train_df):
    """
    Split complete engines between training and validation.

    This prevents windows from the same engine appearing in both sets.
    """

    unit_ids = np.array(sorted(train_df["unit"].unique()))

    random_generator = np.random.default_rng(RANDOM_SEED)
    random_generator.shuffle(unit_ids)

    validation_count = max(
        1,
        int(round(len(unit_ids) * VALIDATION_FRACTION))
    )

    validation_units = unit_ids[:validation_count]
    training_units = unit_ids[validation_count:]

    return training_units, validation_units


def create_training_windows(df, unit_ids, feature_names):
    """Create overlapping sequence windows for complete engine histories."""

    windows = []
    targets = []
    window_units = []

    selected_df = df[df["unit"].isin(unit_ids)]

    for unit_id, unit_df in selected_df.groupby("unit"):
        unit_df = unit_df.sort_values("time")

        feature_values = unit_df[feature_names].to_numpy(dtype=np.float32)
        rul_values = unit_df["RUL"].to_numpy(dtype=np.float32)

        for end_index in range(WINDOW_SIZE - 1, len(unit_df)):
            start_index = end_index - WINDOW_SIZE + 1

            windows.append(feature_values[start_index:end_index + 1])
            targets.append(rul_values[end_index])
            window_units.append(unit_id)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(window_units),
    )


def create_test_windows(test_df, rul_df, feature_names):
    """
    Create one final sequence per test engine.

    The official C-MAPSS test target represents RUL after the last
    observed cycle, so evaluation should use one prediction per engine.
    """

    windows = []
    targets = []
    test_units = []

    unit_ids = np.array(sorted(test_df["unit"].unique()))
    rul_values = rul_df.iloc[:, 0].to_numpy(dtype=np.float32)

    if len(unit_ids) != len(rul_values):
        raise ValueError(
            "Number of test engines does not match number of RUL targets."
        )

    rul_by_unit = dict(zip(unit_ids, rul_values))

    for unit_id, unit_df in test_df.groupby("unit"):
        unit_df = unit_df.sort_values("time")
        feature_values = unit_df[feature_names].to_numpy(dtype=np.float32)

        if len(feature_values) >= WINDOW_SIZE:
            final_window = feature_values[-WINDOW_SIZE:]
        else:
            padding_count = WINDOW_SIZE - len(feature_values)

            padded_values = np.pad(
                feature_values,
                ((padding_count, 0), (0, 0)),
                mode="edge",
            )

            final_window = padded_values

        windows.append(final_window)
        targets.append(min(rul_by_unit[unit_id], RUL_CAP))
        test_units.append(unit_id)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(test_units),
    )


def preprocess_dataset(fd_id):
    print(f"\nProcessing {fd_id}...")

    train_df, test_df, rul_df = load_raw_data(fd_id)
    train_df = add_training_rul(train_df)

    feature_names = [
        column
        for column in train_df.columns
        if column not in ["unit", "time", "RUL"]
    ]

    training_units, validation_units = split_training_units(train_df)
    
    # StandardScaler produces decimal values, so all feature columns
    # must use a floating-point datatype.
    feature_dtypes = {
        feature: np.float64
        for feature in feature_names
    }

    train_df = train_df.astype(feature_dtypes)
    test_df = test_df.astype(feature_dtypes)
    # Fit the scaler only on training engines to prevent leakage.
    scaler = StandardScaler()

    scaler.fit(
        train_df.loc[
            train_df["unit"].isin(training_units),
            feature_names,
        ]
    )

    train_df.loc[:, feature_names] = scaler.transform(
        train_df[feature_names]
    )

    test_df.loc[:, feature_names] = scaler.transform(
        test_df[feature_names]
    )

    X_train, y_train, train_window_units = create_training_windows(
        train_df,
        training_units,
        feature_names,
    )

    X_val, y_val, validation_window_units = create_training_windows(
        train_df,
        validation_units,
        feature_names,
    )

    X_test, y_test, test_units = create_test_windows(
        test_df,
        rul_df,
        feature_names,
    )

    output_file = os.path.join(
        PROCESSED_DIR,
        f"{fd_id}.pkl",
    )

    joblib.dump(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "train_units": training_units,
            "validation_units": validation_units,
            "train_window_units": train_window_units,
            "validation_window_units": validation_window_units,
            "test_units": test_units,
            "feature_names": feature_names,
            "scaler": scaler,
            "window_size": WINDOW_SIZE,
            "rul_cap": RUL_CAP,
        },
        output_file,
    )

    print(f"Training:   X={X_train.shape}, y={y_train.shape}")
    print(f"Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"Testing:    X={X_test.shape}, y={y_test.shape}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    for dataset_id in ["FD001", "FD002", "FD003", "FD004"]:
        preprocess_dataset(dataset_id)