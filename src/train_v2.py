import argparse
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DATA_DIR = "data/processed_v2"
MODELS_DIR = "models_v2"
RESULTS_DIR = "results_v2"

RANDOM_SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_reproducibility(seed=RANDOM_SEED):
    """Set random seeds for reproducible experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data(dataset):
    """Load the leakage-safe preprocessed dataset."""

    dataset_path = os.path.join(DATA_DIR, f"{dataset}.pkl")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}\n"
            "Run preprocessing_v2.py first."
        )

    data = joblib.load(dataset_path)

    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


def build_lstm(input_shape):
    """Build the LSTM regression model."""

    inputs = keras.Input(shape=input_shape)

    x = layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.20,
    )(inputs)

    x = layers.LSTM(
        32,
        dropout=0.20,
    )(x)

    x = layers.Dense(
        32,
        activation="relu",
    )(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(
        1,
        activation="softplus",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="machineguard_lstm",
    )


def build_gru(input_shape):
    """Build the GRU regression model."""

    inputs = keras.Input(shape=input_shape)

    x = layers.GRU(
        64,
        return_sequences=True,
        dropout=0.20,
    )(inputs)

    x = layers.GRU(
        32,
        dropout=0.20,
    )(x)

    x = layers.Dense(
        32,
        activation="relu",
    )(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(
        1,
        activation="softplus",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="machineguard_gru",
    )


def transformer_encoder(inputs):
    """Create one lightweight Transformer encoder block."""

    x = layers.LayerNormalization()(inputs)

    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=16,
        dropout=0.10,
    )(x, x)

    x = layers.Add()([inputs, attention_output])
    normalized_x = layers.LayerNormalization()(x)

    feed_forward = layers.Dense(
        64,
        activation="relu",
    )(normalized_x)

    feed_forward = layers.Dropout(0.10)(feed_forward)

    feed_forward = layers.Dense(
        inputs.shape[-1],
    )(feed_forward)

    return layers.Add()([x, feed_forward])


def build_transformer(input_shape):
    """Build the Transformer regression model."""

    inputs = keras.Input(shape=input_shape)

    x = transformer_encoder(inputs)
    x = transformer_encoder(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        64,
        activation="relu",
    )(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(
        1,
        activation="softplus",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="machineguard_transformer",
    )


def build_model(model_name, input_shape):
    """Select and compile the requested model."""

    if model_name == "lstm":
        model = build_lstm(input_shape)

    elif model_name == "gru":
        model = build_gru(input_shape)

    elif model_name == "transformer":
        model = build_transformer(input_shape)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.Huber(),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


def train_model(
    model_name,
    dataset,
    epochs=100,
    batch_size=64,
):
    """Train, validate and save the best model."""

    set_reproducibility()

    (
        X_train,
        y_train,
        X_val,
        y_val,
        _,
        _,
    ) = load_data(dataset)

    print(f"Training {model_name} on {dataset}")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    model = build_model(
        model_name=model_name,
        input_shape=X_train.shape[1:],
    )

    model.summary()

    model_path = os.path.join(
        MODELS_DIR,
        f"{model_name}_{dataset}.keras",
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    model.save(model_path)

    history_path = os.path.join(
        RESULTS_DIR,
        f"history_{model_name}_{dataset}.csv",
    )

    pd.DataFrame(history.history).to_csv(
        history_path,
        index=False,
    )

    print(f"Best model saved to {model_path}")
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["lstm", "gru", "transformer"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["FD001", "FD002", "FD003", "FD004"],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    arguments = parser.parse_args()

    train_model(
        model_name=arguments.model,
        dataset=arguments.dataset,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
    )