from generate_dataset import handle_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt


def main():
    print("Program start")
    handle_dataset("./dataset.csv")
    df = pd.read_csv("./dataset.csv")

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

    # model = keras.Sequential([
    #     keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, 4)),
    #     keras.layers.LSTM(64),
    #     keras.layers.Dense(2)  # Output joint angles
    # ])

    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(4,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(2),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae", "accuracy"])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    # Evaluate the model
    loss, mae, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    # Plot training & validation loss and MAE

    # plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
    # Save the model
    model.save("robot_kinematics_model.keras")
    print("Model saved successfully")


if __name__ == "__main__":
    main()
