from generate_dataset import handle_dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

def create_sequences(input_positions, input_joints, output_joints, sequence_length=5):
    """Convert data into sequences for RNN training"""
    X = []
    y = []
    
    # For each possible sequence in our data
    for i in range(len(input_positions) - sequence_length + 1):
        # Create a sequence of the specified length
        seq = []
        for j in range(sequence_length):
            # Each sequence element has position and joint angles
            if i+j < len(input_positions):  # Ensure we don't go out of bounds
                seq.append([*input_positions[i+j], *input_joints[i+j]])
        
        # Only add complete sequences
        if len(seq) == sequence_length:
            X.append(seq)
            y.append(output_joints[i + sequence_length - 1])  # Target is the output joint at the end of sequence
    
    return np.array(X), np.array(y)

def main():
    print("Program start")
    handle_dataset("./dataset.csv")
    df = pd.read_csv('./dataset.csv')
    print(df)
    
    # Extract data from dataframe
    # Assuming columns contain string representations of tuples
    input_positions = df['input position'].apply(eval).tolist()
    input_joints = df['input joint'].apply(eval).tolist()
    output_joints = df['output joint'].apply(eval).tolist()
    
    # Create sequences for RNN
    sequence_length = 5  # Consider 5 past steps
    X, y = create_sequences(input_positions, input_joints, output_joints, sequence_length)
    
    print("Sequence data shape:", X.shape)
    print("Target data shape:", y.shape)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create an RNN model
    model = keras.Sequential([
        # Input shape: (sequence_length, features_per_timestep)
        # Each timestep has 4 features: x, y, θ1, θ2
        keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, 4)),
        keras.layers.LSTM(64),  # Second LSTM layer
        keras.layers.Dense(32, activation='relu'),  # Optional Dense layer
        keras.layers.Dense(2)  # Output: θ1, θ2
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    
    # Add early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    loss, mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    
    # Save the model
    model.save('robot_kinematics_rnn_model.keras')
    print("RNN Model saved successfully")

if __name__=="__main__":
    main()