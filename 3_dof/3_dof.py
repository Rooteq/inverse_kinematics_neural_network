from generate_3dof import handle_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt


def main():
    print("Program start")

    handle_dataset("./3dof_dataset.csv")

    print("Loading dataset...")
    df = pd.read_csv("./3dof_dataset.csv")
    print("Dataset loaded. Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    try:

        input_positions_x = df["position_x"].values
        input_positions_y = df["position_y"].values

        input_joints_1 = df["input_joint_1"].values
        input_joints_2 = df["input_joint_2"].values
        input_joints_3 = df["input_joint_3"].values

        output_joints_1 = df["output_joint_1"].values
        output_joints_2 = df["output_joint_2"].values
        output_joints_3 = df["output_joint_3"].values

        X = np.column_stack(
            (
                input_positions_x,
                input_positions_y,
                input_joints_1,
                input_joints_2,
                input_joints_3,
            )
        )

        y = np.column_stack((output_joints_1, output_joints_2, output_joints_3))

    except KeyError:

        print("Using original CSV format with string parsing...")

        input_positions = df["input position"].apply(eval).tolist()
        input_joints = df["input joint"].apply(eval).tolist()
        output_joints = df["output joint"].apply(eval).tolist()

        X = []
        for pos, joint in zip(input_positions, input_joints):
            # pos is (x, y) and joint is (θ1, θ2, θ3)
            x_sample = list(pos) + list(joint)
            X.append(x_sample)

        X = np.array(X)
        y = np.array(output_joints)

    print("Input shape (X):", X.shape)
    print("Output shape (y):", y.shape)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5)
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
            keras.layers.Dense(128, activation="relu", input_shape=(5,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(3),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    model.summary()

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
    print(f"Validation Loss (MSE): {loss:.6f}")
    print(f"Validation MAE: {mae:.6f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("3dof_training_metrics.png")
    plt.show()

    num_test_samples = min(5, len(X_val))
    test_samples = X_val[:num_test_samples]
    true_values = y_val[:num_test_samples]
    predictions = model.predict(test_samples)

    print("\nSample Predictions:")
    for i in range(num_test_samples):
        print(
            f"Input: Position ({test_samples[i][0]:.2f}, {test_samples[i][1]:.2f}), "
            f"Input Joints ({test_samples[i][2]:.2f}, {test_samples[i][3]:.2f}, {test_samples[i][4]:.2f})"
        )
        print(
            f"True Joint Angles: ({true_values[i][0]:.2f}, {true_values[i][1]:.2f}, {true_values[i][2]:.2f})"
        )
        print(
            f"Predicted Joint Angles: ({predictions[i][0]:.2f}, {predictions[i][1]:.2f}, {predictions[i][2]:.2f})"
        )
        print(
            f"Absolute Error: ({abs(true_values[i][0]-predictions[i][0]):.4f}, "
            f"{abs(true_values[i][1]-predictions[i][1]):.4f}, "
            f"{abs(true_values[i][2]-predictions[i][2]):.4f})"
        )
        print()

    model.save("3dof_model.keras")
    print("Model saved successfully as 3dof_model.keras")


if __name__ == "__main__":
    main()
