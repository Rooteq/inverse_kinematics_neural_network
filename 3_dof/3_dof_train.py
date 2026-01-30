import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras


DATASET_PATH = "3dof_dataset.csv"
MODEL_PATH = "3dof_model.keras"


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    
    input_positions = df["input position"].apply(eval).tolist()
    input_joints = df["input joint"].apply(eval).tolist()
    output_joints = df["output joint"].apply(eval).tolist()

    X = []
    for pos, joint in zip(input_positions, input_joints):
        x_sample = list(pos) + list(joint)
        X.append(x_sample)

    X = np.array(X)
    y = np.array(output_joints)
    
    return X, y


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(5,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(3),
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_mae",
        min_delta=0.0005,
        patience=50,
        verbose=0,
        mode="min",
        restore_best_weights=True,
    )
    
    model = create_model()
    
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping],
    )
    
    loss, mae = model.evaluate(X_val, y_val, verbose=0)
    
    return model, loss, mae


def main():
    if not Path(DATASET_PATH).exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Run 3dof_generate.py first to generate the dataset")
        sys.exit(1)
    
    X, y = load_dataset()
    model, loss, mae = train_model(X, y)
    
    model.save(MODEL_PATH)
    print(f"Model trained: Loss={loss:.6f}, MAE={mae:.6f}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
