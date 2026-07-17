import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras


DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed_v2"
MODELS_DIR = Path("models_v2")
RESULTS_DIR = Path("results_v2")
TRAJECTORY_DIR = RESULTS_DIR / "trajectory_diagnostics"

TRAJECTORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def find_cmapss_file(
    file_prefix,
    dataset,
):
    """
    Locate a NASA C-MAPSS raw-data file.

    Several common repository folder structures are supported.
    """

    filename_candidates = [
        f"{file_prefix}_{dataset}.txt",
        f"{file_prefix}_{dataset}.csv",
    ]

    directory_candidates = [
        DATA_DIR / "raw",
        DATA_DIR / "raw" / "CMAPSSData",
        DATA_DIR / "CMAPSSData",
        DATA_DIR,
        Path("CMAPSSData"),
        Path("."),
    ]

    for directory in directory_candidates:
        for filename in filename_candidates:
            candidate_path = directory / filename

            if candidate_path.exists():
                return candidate_path

    searched_paths = [
        str(directory / filename)
        for directory in directory_candidates
        for filename in filename_candidates
    ]

    raise FileNotFoundError(
        f"Could not locate {file_prefix}_{dataset}.\n"
        "Searched:\n"
        + "\n".join(searched_paths)
    )


def load_processed_metadata(
    dataset,
):
    """Load scaler, feature names and sequence configuration."""

    processed_path = PROCESSED_DIR / f"{dataset}.pkl"

    if not processed_path.exists():
        raise FileNotFoundError(
            "Processed dataset not found: "
            f"{processed_path}"
        )

    processed_data = joblib.load(processed_path)

    required_keys = [
        "feature_names",
        "scaler",
        "window_size",
        "rul_cap",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in processed_data
    ]

    if missing_keys:
        raise KeyError(
            "Processed dataset is missing keys: "
            f"{missing_keys}"
        )

    return {
        "feature_names": list(
            processed_data["feature_names"]
        ),
        "scaler": processed_data["scaler"],
        "window_size": int(
            processed_data["window_size"]
        ),
        "rul_cap": float(
            processed_data["rul_cap"]
        ),
        "processed_path": processed_path,
    }


def load_model(
    model_name,
    dataset,
):
    """Load a trained sequence model."""

    model_path = (
        MODELS_DIR
        / f"{model_name}_{dataset}.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = keras.models.load_model(
        model_path
    )

    return model, model_path


def read_test_data(
    dataset,
    feature_names,
):
    """Read the raw NASA C-MAPSS test history."""

    test_path = find_cmapss_file(
        file_prefix="test",
        dataset=dataset,
    )

    raw_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    expected_columns = 2 + len(
        feature_names
    )

    if raw_df.shape[1] < expected_columns:
        raise ValueError(
            f"Expected at least {expected_columns} columns "
            f"in {test_path}, but found "
            f"{raw_df.shape[1]}."
        )

    raw_df = raw_df.iloc[
        :, :expected_columns
    ].copy()

    raw_df.columns = [
        "unit",
        "cycle",
        *feature_names,
    ]

    raw_df["unit"] = raw_df[
        "unit"
    ].astype(int)

    raw_df["cycle"] = raw_df[
        "cycle"
    ].astype(int)

    raw_df[feature_names] = raw_df[
        feature_names
    ].astype(float)

    raw_df = raw_df.sort_values(
        by=[
            "unit",
            "cycle",
        ]
    ).reset_index(
        drop=True
    )

    return raw_df, test_path


def load_final_rul_labels(
    dataset,
    test_units,
):
    """
    Load final observed-cycle RUL labels.

    The script can still run without this file. Actual RUL is used
    only for retrospective evaluation and plotting.
    """

    try:
        rul_path = find_cmapss_file(
            file_prefix="RUL",
            dataset=dataset,
        )

    except FileNotFoundError:
        return None, None

    rul_df = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    rul_values = (
        rul_df
        .iloc[:, 0]
        .astype(float)
        .to_numpy()
    )

    sorted_units = np.sort(
        np.unique(test_units)
    )

    if len(rul_values) < len(sorted_units):
        raise ValueError(
            "The RUL file contains fewer labels than "
            "the number of test engines."
        )

    final_rul_map = {
        int(unit): float(rul_values[index])
        for index, unit in enumerate(
            sorted_units
        )
    }

    return final_rul_map, rul_path


def scale_test_features(
    raw_df,
    feature_names,
    scaler,
):
    """Scale raw features with the training-only scaler."""

    scaled_df = raw_df[
        [
            "unit",
            "cycle",
        ]
    ].copy()

    scaled_features = scaler.transform(
        raw_df[feature_names]
    )

    scaled_df[feature_names] = (
        scaled_features.astype(
            np.float32
        )
    )

    return scaled_df


def build_edge_padded_window(
    feature_array,
    end_index,
    window_size,
):
    """
    Build a fixed-length sequence ending at end_index.

    Early engine history is padded with the first available cycle,
    matching the V2 test-window preprocessing strategy.
    """

    start_index = (
        end_index
        - window_size
        + 1
    )

    if start_index >= 0:
        return feature_array[
            start_index:end_index + 1
        ]

    available_window = feature_array[
        :end_index + 1
    ]

    padding_length = -start_index

    padding = np.repeat(
        feature_array[0:1],
        repeats=padding_length,
        axis=0,
    )

    return np.concatenate(
        [
            padding,
            available_window,
        ],
        axis=0,
    )


def build_history_windows(
    scaled_df,
    feature_names,
    window_size,
):
    """Build one sequence window for every observed engine cycle."""

    windows = []
    metadata_rows = []

    for unit, engine_df in scaled_df.groupby(
        "unit",
        sort=True,
    ):
        engine_df = engine_df.sort_values(
            by="cycle"
        ).reset_index(
            drop=True
        )

        feature_array = engine_df[
            feature_names
        ].to_numpy(
            dtype=np.float32
        )

        cycles = engine_df[
            "cycle"
        ].to_numpy(
            dtype=int
        )

        for end_index in range(
            len(engine_df)
        ):
            window = build_edge_padded_window(
                feature_array=feature_array,
                end_index=end_index,
                window_size=window_size,
            )

            windows.append(window)

            metadata_rows.append(
                {
                    "unit": int(unit),
                    "cycle": int(
                        cycles[end_index]
                    ),
                    "observed_cycle_index": int(
                        end_index + 1
                    ),
                }
            )

    X_history = np.asarray(
        windows,
        dtype=np.float32,
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    return X_history, metadata_df


def generate_history_predictions(
    model,
    X_history,
    rul_cap,
):
    """Generate deterministic RUL predictions for all history windows."""

    predictions = model.predict(
        X_history,
        batch_size=256,
        verbose=1,
    ).reshape(-1)

    return np.clip(
        predictions,
        0.0,
        rul_cap,
    )


def add_actual_rul_trajectory(
    trajectory_df,
    final_rul_map,
    rul_cap,
):
    """
    Add retrospective true-RUL trajectories.

    For a test engine:
    earlier true RUL = final supplied RUL
    + cycles remaining until the last observation.
    """

    trajectory_df = trajectory_df.copy()

    if final_rul_map is None:
        trajectory_df[
            "actual_rul"
        ] = np.nan

        trajectory_df[
            "actual_rul_capped"
        ] = np.nan

        trajectory_df[
            "prediction_error"
        ] = np.nan

        trajectory_df[
            "absolute_error"
        ] = np.nan

        return trajectory_df

    trajectory_df[
        "actual_rul"
    ] = np.nan

    for unit, engine_df in trajectory_df.groupby(
        "unit",
        sort=False,
    ):
        maximum_cycle = int(
            engine_df["cycle"].max()
        )

        raw_final_rul = float(
            final_rul_map[int(unit)]
        )

        engine_actual_rul = (
            raw_final_rul
            + maximum_cycle
            - engine_df[
                "cycle"
            ].to_numpy(
                dtype=float
            )
        )

        trajectory_df.loc[
            engine_df.index,
            "actual_rul",
        ] = engine_actual_rul

    trajectory_df[
        "actual_rul_capped"
    ] = np.clip(
        trajectory_df["actual_rul"],
        0.0,
        rul_cap,
    )

    trajectory_df[
        "prediction_error"
    ] = (
        trajectory_df[
            "predicted_rul"
        ]
        - trajectory_df[
            "actual_rul_capped"
        ]
    )

    trajectory_df[
        "absolute_error"
    ] = np.abs(
        trajectory_df[
            "prediction_error"
        ]
    )

    return trajectory_df


def calculate_linear_slope(
    cycles,
    predictions,
):
    """Calculate prediction change per observed cycle."""

    cycles = np.asarray(
        cycles,
        dtype=float,
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if len(predictions) < 2:
        return 0.0

    slope, _ = np.polyfit(
        cycles,
        predictions,
        deg=1,
    )

    return float(slope)


def calculate_residual_std(
    cycles,
    predictions,
):
    """Measure deviation from a linear recent trajectory."""

    cycles = np.asarray(
        cycles,
        dtype=float,
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if len(predictions) < 3:
        return 0.0

    slope, intercept = np.polyfit(
        cycles,
        predictions,
        deg=1,
    )

    fitted_predictions = (
        slope * cycles
        + intercept
    )

    residuals = (
        predictions
        - fitted_predictions
    )

    return float(
        np.std(residuals)
    )


def classify_trajectory(
    final_predicted_rul,
    recent_slope,
    recent_predicted_drop,
    recent_range,
    violation_rate,
    large_jump_count,
    residual_std,
):
    """
    Assign a transparent trajectory diagnostic.

    These are heuristic diagnostic flags, not learned maintenance
    decisions and not tuned using test targets.
    """

    if (
        large_jump_count >= 2
        or residual_std > 12
        or violation_rate > 0.60
    ):
        return "Unstable trajectory"

    if (
        final_predicted_rul >= 90
        and recent_predicted_drop < 8
        and recent_slope > -0.40
    ):
        return "High-RUL plateau — review"

    if (
        final_predicted_rul >= 80
        and recent_slope > 0.10
    ):
        return "RUL increasing despite ageing"

    if (
        recent_slope < -1.75
        or recent_predicted_drop > 30
    ):
        return "Rapid degradation signal"

    if (
        recent_slope <= -0.25
        and violation_rate <= 0.45
    ):
        return "Consistent degradation"

    if (
        abs(recent_slope) < 0.25
        and recent_range < 12
    ):
        return "Weak degradation / plateau"

    return "Mixed trajectory"


def build_engine_summary(
    trajectory_df,
    recent_window,
    increase_tolerance,
    jump_threshold,
):
    """Calculate trajectory diagnostics for every test engine."""

    summary_rows = []

    for unit, engine_df in trajectory_df.groupby(
        "unit",
        sort=True,
    ):
        engine_df = engine_df.sort_values(
            by="cycle"
        ).reset_index(
            drop=True
        )

        recent_df = engine_df.tail(
            recent_window
        )

        recent_cycles = recent_df[
            "cycle"
        ].to_numpy(
            dtype=float
        )

        recent_predictions = recent_df[
            "predicted_rul"
        ].to_numpy(
            dtype=float
        )

        all_cycles = engine_df[
            "cycle"
        ].to_numpy(
            dtype=float
        )

        all_predictions = engine_df[
            "predicted_rul"
        ].to_numpy(
            dtype=float
        )

        recent_differences = np.diff(
            recent_predictions
        )

        if len(recent_differences) == 0:
            violation_rate = 0.0
            flat_step_rate = 1.0
            large_jump_count = 0
            largest_jump = 0.0

        else:
            violation_rate = float(
                np.mean(
                    recent_differences
                    > increase_tolerance
                )
            )

            flat_step_rate = float(
                np.mean(
                    np.abs(
                        recent_differences
                    )
                    < 0.5
                )
            )

            large_jump_count = int(
                np.sum(
                    np.abs(
                        recent_differences
                    )
                    > jump_threshold
                )
            )

            largest_jump = float(
                np.max(
                    np.abs(
                        recent_differences
                    )
                )
            )

        recent_slope = calculate_linear_slope(
            cycles=recent_cycles,
            predictions=recent_predictions,
        )

        all_history_slope = calculate_linear_slope(
            cycles=all_cycles,
            predictions=all_predictions,
        )

        recent_residual_std = calculate_residual_std(
            cycles=recent_cycles,
            predictions=recent_predictions,
        )

        recent_predicted_drop = float(
            recent_predictions[0]
            - recent_predictions[-1]
        )

        recent_expected_drop = float(
            max(
                len(recent_predictions) - 1,
                0,
            )
        )

        if recent_expected_drop > 0:
            recent_decline_ratio = float(
                recent_predicted_drop
                / recent_expected_drop
            )

        else:
            recent_decline_ratio = 0.0

        recent_range = float(
            np.ptp(recent_predictions)
        )

        final_predicted_rul = float(
            engine_df[
                "predicted_rul"
            ].iloc[-1]
        )

        trajectory_flag = classify_trajectory(
            final_predicted_rul=final_predicted_rul,
            recent_slope=recent_slope,
            recent_predicted_drop=recent_predicted_drop,
            recent_range=recent_range,
            violation_rate=violation_rate,
            large_jump_count=large_jump_count,
            residual_std=recent_residual_std,
        )

        summary_row = {
            "unit": int(unit),
            "observed_cycles": int(
                len(engine_df)
            ),
            "recent_window_used": int(
                len(recent_df)
            ),
            "initial_predicted_rul": float(
                engine_df[
                    "predicted_rul"
                ].iloc[0]
            ),
            "final_predicted_rul": (
                final_predicted_rul
            ),
            "all_history_slope": (
                all_history_slope
            ),
            "recent_slope": (
                recent_slope
            ),
            "recent_degradation_rate": float(
                -recent_slope
            ),
            "recent_predicted_drop": (
                recent_predicted_drop
            ),
            "recent_expected_drop": (
                recent_expected_drop
            ),
            "recent_decline_ratio": (
                recent_decline_ratio
            ),
            "recent_prediction_range": (
                recent_range
            ),
            "recent_prediction_std": float(
                np.std(
                    recent_predictions
                )
            ),
            "recent_residual_std": (
                recent_residual_std
            ),
            "monotonicity_violation_rate": (
                violation_rate
            ),
            "flat_step_rate": (
                flat_step_rate
            ),
            "large_jump_count": (
                large_jump_count
            ),
            "largest_single_cycle_jump": (
                largest_jump
            ),
            "trajectory_flag": (
                trajectory_flag
            ),
        }

        if engine_df[
            "actual_rul_capped"
        ].notna().any():
            raw_final_actual_rul = float(
                engine_df[
                    "actual_rul"
                ].iloc[-1]
            )

            final_actual_rul = float(
                engine_df[
                    "actual_rul_capped"
                ].iloc[-1]
            )

            final_prediction_error = (
                final_predicted_rul
                - final_actual_rul
            )

            final_absolute_error = abs(
                final_prediction_error
            )

            summary_row[
                "raw_final_actual_rul"
            ] = raw_final_actual_rul

            summary_row[
                "final_actual_rul"
            ] = final_actual_rul

            summary_row[
                "final_prediction_error"
            ] = final_prediction_error

            summary_row[
                "final_absolute_error"
            ] = final_absolute_error

        summary_rows.append(
            summary_row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    return summary_df.sort_values(
        by=[
            "trajectory_flag",
            "unit",
        ]
    ).reset_index(
        drop=True
    )


def build_flag_summary(
    engine_summary_df,
):
    """Summarize the diagnostic categories."""

    flag_summary_df = (
        engine_summary_df[
            "trajectory_flag"
        ]
        .value_counts()
        .rename_axis(
            "trajectory_flag"
        )
        .reset_index(
            name="engine_count"
        )
    )

    flag_summary_df[
        "percentage"
    ] = (
        flag_summary_df[
            "engine_count"
        ]
        / len(engine_summary_df)
        * 100
    )

    if (
        "final_absolute_error"
        in engine_summary_df.columns
    ):
        error_summary = (
            engine_summary_df.groupby(
                "trajectory_flag"
            )[
                "final_absolute_error"
            ]
            .mean()
            .rename(
                "average_final_absolute_error"
            )
            .reset_index()
        )

        flag_summary_df = flag_summary_df.merge(
            error_summary,
            on="trajectory_flag",
            how="left",
        )

    return flag_summary_df


def plot_engine_trajectory(
    trajectory_df,
    unit,
    model_name,
    dataset,
    recent_window,
):
    """Plot one engine's predicted RUL across its full history."""

    engine_df = trajectory_df[
        trajectory_df["unit"] == unit
    ].sort_values(
        by="cycle"
    )

    if engine_df.empty:
        return None

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        engine_df["cycle"],
        engine_df["predicted_rul"],
        linewidth=2.2,
        label=(
            f"{model_name.upper()} "
            "predicted RUL"
        ),
    )

    if engine_df[
        "actual_rul_capped"
    ].notna().any():
        plt.plot(
            engine_df["cycle"],
            engine_df[
                "actual_rul_capped"
            ],
            linewidth=2.5,
            linestyle="--",
            label="Actual capped RUL",
        )

    recent_df = engine_df.tail(
        recent_window
    )

    plt.axvspan(
        recent_df["cycle"].iloc[0],
        recent_df["cycle"].iloc[-1],
        alpha=0.12,
        label=(
            f"Recent {len(recent_df)} cycles"
        ),
    )

    plt.xlabel(
        "Observed engine cycle"
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.title(
        f"RUL Trajectory — Unit {unit} — "
        f"{model_name.upper()} {dataset}"
    )

    plt.legend()

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = (
        TRAJECTORY_DIR
        / (
            f"trajectory_{model_name}_"
            f"{dataset}_unit_{unit}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_selected_final_predictions(
    engine_summary_df,
    selected_units,
    model_name,
    dataset,
):
    """Plot final predictions and capped actual RUL for selected engines."""

    plot_df = engine_summary_df[
        engine_summary_df[
            "unit"
        ].isin(selected_units)
    ].copy()

    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(
        by="unit"
    ).reset_index(
        drop=True
    )

    positions = np.arange(
        len(plot_df)
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        positions,
        plot_df[
            "final_predicted_rul"
        ],
        width=0.55,
        label=(
            f"{model_name.upper()} "
            "final prediction"
        ),
    )

    if (
        "final_actual_rul"
        in plot_df.columns
    ):
        plt.scatter(
            positions,
            plot_df[
                "final_actual_rul"
            ],
            marker="x",
            s=100,
            label="Actual capped final RUL",
        )

    plt.xticks(
        positions,
        [
            f"Unit {unit}"
            for unit in plot_df[
                "unit"
            ]
        ],
    )

    plt.xlabel(
        "Test engine"
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.title(
        "Final Predictions for Selected Engines — "
        f"{model_name.upper()} {dataset}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = (
        TRAJECTORY_DIR
        / (
            "selected_final_predictions_"
            f"{model_name}_{dataset}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def print_selected_engines(
    engine_summary_df,
    selected_units,
):
    """Print selected engine trajectory diagnostics."""

    selected_df = engine_summary_df[
        engine_summary_df[
            "unit"
        ].isin(selected_units)
    ].copy()

    if selected_df.empty:
        print(
            "No selected engines were found."
        )
        return

    columns = [
        "unit",
        "observed_cycles",
        "final_predicted_rul",
    ]

    optional_columns = [
        "raw_final_actual_rul",
        "final_actual_rul",
        "final_prediction_error",
        "final_absolute_error",
    ]

    columns.extend(
        [
            column
            for column in optional_columns
            if column in selected_df.columns
        ]
    )

    columns.extend(
        [
            "recent_slope",
            "recent_predicted_drop",
            "recent_decline_ratio",
            "recent_prediction_range",
            "monotonicity_violation_rate",
            "large_jump_count",
            "trajectory_flag",
        ]
    )

    print(
        selected_df[
            columns
        ]
        .sort_values(
            by="unit"
        )
        .to_string(
            index=False
        )
    )


def run_trajectory_diagnostics(
    model_name,
    dataset,
    selected_units,
    recent_window,
    increase_tolerance,
    jump_threshold,
):
    """Run full-history temporal consistency diagnostics."""

    metadata = load_processed_metadata(
        dataset=dataset
    )

    model, model_path = load_model(
        model_name=model_name,
        dataset=dataset,
    )

    raw_df, test_path = read_test_data(
        dataset=dataset,
        feature_names=metadata[
            "feature_names"
        ],
    )

    final_rul_map, rul_path = load_final_rul_labels(
        dataset=dataset,
        test_units=raw_df[
            "unit"
        ].to_numpy(),
    )

    scaled_df = scale_test_features(
        raw_df=raw_df,
        feature_names=metadata[
            "feature_names"
        ],
        scaler=metadata["scaler"],
    )

    X_history, trajectory_df = build_history_windows(
        scaled_df=scaled_df,
        feature_names=metadata[
            "feature_names"
        ],
        window_size=metadata[
            "window_size"
        ],
    )

    print(
        f"History windows: {len(X_history)}"
    )

    print(
        "Window shape: "
        f"{X_history.shape[1:]}"
    )

    predictions = generate_history_predictions(
        model=model,
        X_history=X_history,
        rul_cap=metadata["rul_cap"],
    )

    trajectory_df[
        "predicted_rul"
    ] = predictions

    trajectory_df = add_actual_rul_trajectory(
        trajectory_df=trajectory_df,
        final_rul_map=final_rul_map,
        rul_cap=metadata["rul_cap"],
    )

    engine_summary_df = build_engine_summary(
        trajectory_df=trajectory_df,
        recent_window=recent_window,
        increase_tolerance=increase_tolerance,
        jump_threshold=jump_threshold,
    )

    flag_summary_df = build_flag_summary(
        engine_summary_df
    )

    trajectory_path = (
        TRAJECTORY_DIR
        / (
            "trajectory_predictions_"
            f"{model_name}_{dataset}.csv"
        )
    )

    summary_path = (
        TRAJECTORY_DIR
        / (
            "trajectory_summary_"
            f"{model_name}_{dataset}.csv"
        )
    )

    flag_summary_path = (
        TRAJECTORY_DIR
        / (
            "trajectory_flag_summary_"
            f"{model_name}_{dataset}.csv"
        )
    )

    trajectory_df.to_csv(
        trajectory_path,
        index=False,
    )

    engine_summary_df.to_csv(
        summary_path,
        index=False,
    )

    flag_summary_df.to_csv(
        flag_summary_path,
        index=False,
    )

    selected_plot_path = (
        plot_selected_final_predictions(
            engine_summary_df=(
                engine_summary_df
            ),
            selected_units=selected_units,
            model_name=model_name,
            dataset=dataset,
        )
    )

    trajectory_plot_paths = []

    for unit in selected_units:
        plot_path = plot_engine_trajectory(
            trajectory_df=trajectory_df,
            unit=unit,
            model_name=model_name,
            dataset=dataset,
            recent_window=recent_window,
        )

        if plot_path is not None:
            trajectory_plot_paths.append(
                plot_path
            )

    print(
        f"Model: {model_path}"
    )

    print(
        "Processed data: "
        f"{metadata['processed_path']}"
    )

    print(
        f"Raw test data: {test_path}"
    )

    if rul_path is None:
        print(
            "RUL labels: Not found — "
            "running diagnostics without actual RUL."
        )

    else:
        print(
            f"RUL labels: {rul_path}"
        )

    print(
        "Test engines: "
        f"{engine_summary_df['unit'].nunique()}"
    )

    print(
        f"Recent window: {recent_window} cycles"
    )

    print(
        "\nTrajectory flag summary"
    )

    print(
        flag_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSelected engine diagnostics"
    )

    print_selected_engines(
        engine_summary_df=engine_summary_df,
        selected_units=selected_units,
    )

    print(
        "\nSaved outputs"
    )

    print(
        f"Cycle predictions: {trajectory_path}"
    )

    print(
        f"Engine summary: {summary_path}"
    )

    print(
        f"Flag summary: {flag_summary_path}"
    )

    if selected_plot_path is not None:
        print(
            f"Selected engines: {selected_plot_path}"
        )

    for plot_path in trajectory_plot_paths:
        print(
            f"Trajectory plot: {plot_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="gru",
        choices=[
            "lstm",
            "gru",
            "transformer",
        ],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "FD001",
            "FD002",
            "FD003",
            "FD004",
        ],
    )

    parser.add_argument(
        "--units",
        type=int,
        nargs="*",
        default=[
            25,
            45,
            67,
            79,
        ],
    )

    parser.add_argument(
        "--recent_window",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--increase_tolerance",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--jump_threshold",
        type=float,
        default=10.0,
    )

    arguments = parser.parse_args()

    if arguments.recent_window < 3:
        raise ValueError(
            "recent_window must be at least 3."
        )

    if arguments.increase_tolerance < 0:
        raise ValueError(
            "increase_tolerance cannot be negative."
        )

    if arguments.jump_threshold <= 0:
        raise ValueError(
            "jump_threshold must be greater than zero."
        )

    run_trajectory_diagnostics(
        model_name=arguments.model,
        dataset=arguments.dataset,
        selected_units=arguments.units,
        recent_window=arguments.recent_window,
        increase_tolerance=(
            arguments.increase_tolerance
        ),
        jump_threshold=(
            arguments.jump_threshold
        ),
    )
