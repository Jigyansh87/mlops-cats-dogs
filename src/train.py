import os
import json
import tensorflow as tf
import mlflow
import mlflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
DATA_DIR = "data/processed"
MODEL_DIR = "models"
METRICS_PATH = "metrics.json"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # MLflow setup
    mlflow.set_experiment("cats_vs_dogs_classification")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        # Data generators
        train_gen = ImageDataGenerator(rescale=1.0 / 255)
        val_gen = ImageDataGenerator(rescale=1.0 / 255)

        train_data = train_gen.flow_from_directory(
            os.path.join(DATA_DIR, "train"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary"
        )

        val_data = val_gen.flow_from_directory(
            os.path.join(DATA_DIR, "val"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary"
        )

        # Build & train model
        model = build_model()
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS
        )

        # Save model
        model_path = os.path.join(MODEL_DIR, "model.h5")
        model.save(model_path)

        # Log model to MLflow
        mlflow.keras.log_model(model, artifact_path="model")

        # Log metrics
        final_val_acc = history.history["val_accuracy"][-1]
        final_val_loss = history.history["val_loss"][-1]

        mlflow.log_metric("val_accuracy", final_val_acc)
        mlflow.log_metric("val_loss", final_val_loss)

        # Save metrics.json for DVC
        metrics = {
            "val_accuracy": float(final_val_acc),
            "val_loss": float(final_val_loss)
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        print("Training completed successfully.")
        print("Validation Accuracy:", final_val_acc)


if __name__ == "__main__":
    main()
