from my_robot_generate import handle_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt

def main():
    print("Program start")
    handle_dataset("./my_robot_dataset.csv")  # Updated path for 3DOF dataset
    
    df = pd.read_csv('./my_robot_dataset.csv')
    input_positions = df['input position'].apply(eval).tolist()
    input_joints = df['input joint'].apply(eval).tolist()
    output_joints = df['output joint'].apply(eval).tolist()
    
    X = []
    for pos, joint in zip(input_positions, input_joints):
        x_sample = list(pos) + list(joint)  # Now includes x, y, z positions and 3 joint angles
        X.append(x_sample)
    
    X = np.array(X)
    y = np.array(output_joints)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Updated model for 3DOF robot
    # Input shape is now 6: 3 for position (x,y,z) and 3 for joint angles
    # Output is 3 joint angles
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(6,)),  # Increased neurons
        keras.layers.Dense(128, activation='relu'),                    # Increased neurons
        keras.layers.Dense(64, activation='relu'),                     # Added layer for complexity
        keras.layers.Dense(3)                                          # Output is 3 joint angles
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,                # Increased epochs for better learning
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate the model
    loss, mae, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Plot training & validation metrics
    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('my_robot_metrics.png')
    plt.show()
    
    # Save the model
    model.save('my_robot_model.keras')
    print("3DOF model saved successfully")

if __name__=="__main__":
    main()