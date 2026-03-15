"""
model_training.py - Train demand forecasting models
=====================================================
Trains 4 Keras deep learning models (LSTM, BiGRU, CNN-LSTM, Attention)
+ 2 classical ML baselines (XGBoost, LightGBM), tracked via MLflow.

Usage: python src/model_training.py
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Keras imports
from keras.models import Sequential, Model
from keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, Bidirectional,
    Input, Concatenate, Reshape
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MLflow (optional — works without it)
try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class DemandModelTrainer:
    """Trains and evaluates multiple demand forecasting models."""

    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.models = {}
        self.metrics = {}
        self.best_model_name = None

    def load_data(self, path="data/processed/features_encoded.csv"):
        """Load feature-engineered data."""
        if not os.path.exists(path):
            print("ERROR: features_encoded.csv not found. Run feature_engineering.py first.")
            exit(1)
        df = pd.read_csv(path)
        print(f"Loaded {len(df):,} rows, {df.shape[1]} features")
        return df

    def prepare_data(self, df, target="units_sold"):
        """Split, scale, and create sequences for Keras models."""
        if target not in df.columns:
            print(f"ERROR: Target column '{target}' not found")
            exit(1)

        X = df.drop(columns=[target])
        y = df[target].values

        # Subsample for CPU training (use 50K rows max to keep training fast)
        max_rows = 50000
        if len(X) > max_rows:
            step = len(X) // max_rows
            X = X.iloc[::step].head(max_rows)
            y = y[::step][:max_rows]
            print(f"  Subsampled to {len(X):,} rows for CPU training")

        # Train/test split (80/20, preserving time order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = X.columns.tolist()

        # Create sequences for Keras models
        self._create_sequences()
        return self

    def _create_sequences(self):
        """Reshape data into 3D sequences for Keras (samples, timesteps, features)."""
        n_features = self.X_train.shape[1]
        seq_len = min(self.sequence_length, self.X_train.shape[0] // 10)
        seq_len = max(seq_len, 5)

        def make_sequences(X, y, length):
            Xs, ys = [], []
            for i in range(length, len(X)):
                Xs.append(X[i - length:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        self.X_train_seq, self.y_train_seq = make_sequences(self.X_train, self.y_train, seq_len)
        self.X_test_seq, self.y_test_seq = make_sequences(self.X_test, self.y_test, seq_len)
        self.seq_len = seq_len
        self.n_features = n_features
        print(f"  Sequences: train={self.X_train_seq.shape}, test={self.X_test_seq.shape}")

    def _get_callbacks(self, model_name):
        """Standard Keras callbacks for all models."""
        os.makedirs("models", exist_ok=True)
        return [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
            ModelCheckpoint(f"models/{model_name}.keras", monitor="val_loss",
                            save_best_only=True, verbose=0),
        ]

    def build_lstm_model(self):
        """LSTM model — captures long-term sequential demand patterns."""
        model = Sequential([
            Input(shape=(self.seq_len, self.n_features)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model

    def build_bigru_model(self):
        """Bidirectional GRU — captures both forward and backward demand trends."""
        model = Sequential([
            Input(shape=(self.seq_len, self.n_features)),
            Bidirectional(GRU(96, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(GRU(48)),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model

    def build_cnn_lstm_model(self):
        """CNN-LSTM hybrid — CNN extracts local patterns, LSTM captures sequence."""
        model = Sequential([
            Input(shape=(self.seq_len, self.n_features)),
            Conv1D(64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            LSTM(64),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model

    def build_attention_model(self):
        """Attention-based model — weighs important timesteps more heavily."""
        inputs = Input(shape=(self.seq_len, self.n_features))
        lstm_out = LSTM(128, return_sequences=True)(inputs)
        lstm_out = Dropout(0.3)(lstm_out)

        # Simple attention mechanism
        attention_weights = Dense(1, activation="tanh")(lstm_out)
        attention_weights = Flatten()(attention_weights)
        attention_weights = Dense(self.seq_len, activation="softmax")(attention_weights)
        attention_weights = Reshape((self.seq_len, 1))(attention_weights)

        context = lstm_out * attention_weights
        context = LSTM(64)(context)
        context = BatchNormalization()(context)
        output = Dense(32, activation="relu")(context)
        output = Dense(1)(output)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model

    def train_keras_model(self, model, name, epochs=50, batch_size=256):
        """Train a Keras model with callbacks and MLflow logging."""
        print(f"\n  Training {name}...")
        start = time.time()

        callbacks = self._get_callbacks(name)
        history = model.fit(
            self.X_train_seq, self.y_train_seq,
            validation_split=0.2,
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=0
        )

        # Evaluate
        y_pred = model.predict(self.X_test_seq, verbose=0).flatten()
        metrics = self._compute_metrics(self.y_test_seq, y_pred)
        metrics["train_time_sec"] = round(time.time() - start, 2)
        metrics["epochs_trained"] = len(history.history["loss"])

        self.models[name] = model
        self.metrics[name] = metrics

        # MLflow logging
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({"model_type": name, "epochs": epochs, "batch_size": batch_size})
                mlflow.log_metrics(metrics)
                mlflow.keras.log_model(model, name)

        self._print_metrics(name, metrics)
        return model

    def train_xgboost(self):
        """Train XGBoost baseline."""
        print("\n  Training XGBoost...")
        start = time.time()
        model = XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        model.fit(self.X_train, self.y_train,
                  eval_set=[(self.X_test, self.y_test)], verbose=False)

        y_pred = model.predict(self.X_test)
        metrics = self._compute_metrics(self.y_test, y_pred)
        metrics["train_time_sec"] = round(time.time() - start, 2)

        self.models["XGBoost"] = model
        self.metrics["XGBoost"] = metrics

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="XGBoost", nested=True):
                mlflow.log_metrics(metrics)

        self._print_metrics("XGBoost", metrics)
        return model

    def train_lightgbm(self):
        """Train LightGBM baseline."""
        print("\n  Training LightGBM...")
        start = time.time()
        model = LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        model.fit(self.X_train, self.y_train,
                  eval_set=[(self.X_test, self.y_test)])

        y_pred = model.predict(self.X_test)
        metrics = self._compute_metrics(self.y_test, y_pred)
        metrics["train_time_sec"] = round(time.time() - start, 2)

        self.models["LightGBM"] = model
        self.metrics["LightGBM"] = metrics

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="LightGBM", nested=True):
                mlflow.log_metrics(metrics)

        self._print_metrics("LightGBM", metrics)
        return model

    def _compute_metrics(self, y_true, y_pred):
        """Compute regression metrics."""
        return {
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            "r2": round(r2_score(y_true, y_pred), 4),
            "mape": round(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100, 2),
        }

    def _print_metrics(self, name, metrics):
        """Print model metrics."""
        print(f"    {name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
              f"R²={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")

    def select_best_model(self):
        """Select best model by R² score."""
        self.best_model_name = max(self.metrics, key=lambda k: self.metrics[k]["r2"])
        best = self.metrics[self.best_model_name]
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name} (R²={best['r2']:.4f})")
        print(f"{'='*60}")

        # Save best model
        os.makedirs("models", exist_ok=True)
        best_model = self.models[self.best_model_name]
        if hasattr(best_model, "save"):  # Keras model
            best_model.save("models/best_model.keras")
        else:  # sklearn/xgboost model
            joblib.dump(best_model, "models/best_model.pkl")

        joblib.dump(self.scaler, "models/scaler.pkl")

        # Save metrics
        with open("models/model_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        print("Saved: models/best_model.keras, models/scaler.pkl, models/model_metrics.json")

    def plot_comparison(self):
        """Generate model comparison chart."""
        os.makedirs("reports/figures", exist_ok=True)
        names = list(self.metrics.keys())
        r2_scores = [self.metrics[n]["r2"] for n in names]
        mae_scores = [self.metrics[n]["mae"] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["#2ecc71" if n == self.best_model_name else "#3498db" for n in names]

        axes[0].barh(names, r2_scores, color=colors)
        axes[0].set_xlabel("R² Score")
        axes[0].set_title("Model Comparison — R²")
        for i, v in enumerate(r2_scores):
            axes[0].text(v + 0.01, i, f"{v:.4f}", va="center")

        axes[1].barh(names, mae_scores, color=colors)
        axes[1].set_xlabel("MAE")
        axes[1].set_title("Model Comparison — MAE")
        for i, v in enumerate(mae_scores):
            axes[1].text(v + 0.1, i, f"{v:.2f}", va="center")

        plt.tight_layout()
        plt.savefig("reports/figures/model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: reports/figures/model_comparison.png")


def run_full_pipeline():
    """Execute the full model training pipeline."""
    print("=" * 60 + "\nMODEL TRAINING PIPELINE\n" + "=" * 60)

    trainer = DemandModelTrainer(sequence_length=30)
    df = trainer.load_data()
    trainer.prepare_data(df)

    # Start MLflow experiment (use temp dir to avoid spaces-in-path bug on Windows)
    if MLFLOW_AVAILABLE:
        import tempfile
        mlruns_dir = os.path.join(tempfile.gettempdir(), "mlruns_retail")
        os.makedirs(mlruns_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")
        mlflow.set_experiment("retail-demand-forecasting")
        parent_run = mlflow.start_run(run_name="full_training")

    # Train Keras models (10 epochs for CPU, EarlyStopping will halt sooner if converged)
    print("\n--- Keras Models ---")
    trainer.train_keras_model(trainer.build_lstm_model(), "LSTM", epochs=10)
    trainer.train_keras_model(trainer.build_bigru_model(), "BiGRU", epochs=10)
    trainer.train_keras_model(trainer.build_cnn_lstm_model(), "CNN_LSTM", epochs=10)
    trainer.train_keras_model(trainer.build_attention_model(), "Attention", epochs=10)

    # Train classical models
    print("\n--- Classical Models ---")
    trainer.train_xgboost()
    trainer.train_lightgbm()

    # Select best & visualize
    trainer.select_best_model()
    trainer.plot_comparison()

    if MLFLOW_AVAILABLE:
        mlflow.end_run()

    return trainer


if __name__ == "__main__":
    run_full_pipeline()
