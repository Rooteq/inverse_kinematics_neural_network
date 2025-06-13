from real_robot_generate import handle_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt


def main():
    print("Program start")
    handle_dataset("./real_robot_dataset.csv")

    df = pd.read_csv("./real_robot_dataset.csv")
    input_positions = df["input position"].apply(eval).tolist()
    input_joints = df["input joint"].apply(eval).tolist()
    output_joints = df["output joint"].apply(eval).tolist()

    X = []
    for pos, joint in zip(input_positions, input_joints):
        x_sample = list(pos) + list(joint)
        X.append(x_sample)

    X = np.array(X)
    y = np.array(output_joints)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_mae",
        min_delta=0.0005,  # Minimum change to qualify as improvement (in radians)
        patience=50,  # Number of epochs with no improvement to wait
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    model = keras.Sequential(
        [
            keras.layers.Dense(128, activation="relu", input_shape=(6,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(3),
        ]
    )

    model.compile(optimizer="adam", loss="log_cosh", metrics=["mae"])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=170,
        batch_size=450,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping],
    )

    loss, mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation MAE: {mae:.4f}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Model MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("real_robot_metrics.png")
    plt.show()

    model.save("real_robot_model.keras")
    print("3DOF model saved successfully")


if __name__ == "__main__":
    main()
