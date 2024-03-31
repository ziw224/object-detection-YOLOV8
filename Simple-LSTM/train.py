import os
import pickle
import numpy as np
import tensorflow as tf
from model import Model
from dataset import DataLoader
import matplotlib.pyplot as plt


def build_model():
    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, 2)),  # Input shape: (batch_size, seq_length, num_features)
        tf.keras.layers.Dense(64, activation='relu'),  # Fully connected layer with ReLU activation
        tf.keras.layers.LSTM(128, return_sequences=True),  # LSTM layer with 128 units and return sequences
        tf.keras.layers.Dense(2)  # Output layer with 2 units (x, y coordinates)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')  # Using mean squared error loss and Adam optimizer

    return model

def train_and_save_predictions():
    # Define the training parameters
    batch_size = 50  # Adjust as needed
    seq_length = 5
    num_epochs = 200 # Adjuest as needed

    # Initialize list to keep track of loss values
    loss_history = []

    # Initialize the data loader object
    data_loader = DataLoader(batch_size=batch_size, seq_length=seq_length)

    # Build the prediction model
    model = build_model()

    # Define the optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.003, decay=0.95)

    # Start training
    for epoch in range(num_epochs):
        # Reset data pointer for each epoch
        data_loader.reset_batch_pointer()

        for batch in range(data_loader.num_batches):
            # Get the next batch of training data
            x_batch, y_batch = data_loader.next_batch()

            # Convert to NumPy arrays
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            # Reshape input data to match the model's input shape
            x_batch = np.reshape(x_batch, (batch_size, seq_length, 2))  # Assuming 2 features

            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(x_batch)

                # Compute loss
                loss = tf.reduce_mean(tf.square(predictions - y_batch))

            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Update weights
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Print and append the loss value to the history
            current_loss = loss.numpy()

            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch + 1}/{data_loader.num_batches}, Loss: {loss.numpy():.4f}")
            loss_history.append(current_loss)

    # After training, plot the training losses
    plt.figure(figsize=(10, 5))  # Optional: Change the figure size
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()
    plt.show()

# After training, save the predictions into pred_results.pkl
    input_data = data_loader.data  # Assuming this is your input data

    # Pad input data to a uniform length
    input_data_padded = tf.keras.preprocessing.sequence.pad_sequences(input_data, padding='post', maxlen=seq_length, dtype='float32')

    pred_results = {
        "predictions": model.predict(input_data_padded),
        "ground_truth": input_data_padded  # Assuming ground truth is the padded input data
    }

    with open("pred_results.pkl", "wb") as f:
        pickle.dump(pred_results, f)

if __name__ == "__main__":
    train_and_save_predictions()