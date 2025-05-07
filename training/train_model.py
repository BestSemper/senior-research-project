import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Masking, Conv1D, GlobalAveragePooling1D, Dense # type: ignore


def parse_dataset(file_path, subframe_length=30):
    """
    Parse the dataset file and extract the labels and sequences.

    Returns:
        labels: A numpy array of labels.
        sequences: A numpy array of sequences.
    """
    video_names = []
    labels = []
    sequences = []
    with open(file_path, "r") as f:
        videos = f.read().strip().split("\n\n\n")
        for video in videos:
            lines = video.split("\n")
            slalom_points = eval(lines[1])
            frames = [[component for keypoint in eval(lines[i]) for component in keypoint] for i in range(2, len(lines))]
            
            # Split frames into subframes of 30 frames each
            subframes = [frames[i-subframe_length:i] for i in range(subframe_length, len(frames))]
            labels += [slalom_points] * (len(frames) - subframe_length)
            sequences += subframes
    sequences = np.array(sequences)
    labels = np.array(labels)
    return labels, sequences


def main():
    subframe_length = 30
    num_epochs = 16
    labels, sequences = parse_dataset("dataset.txt", subframe_length)
    num_features = 34

    # Neural network with two hidden Conv1D layers and a GlobalAveragePooling1D layer
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(subframe_length, num_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    model.summary()

    # # Neural network with LSTM layers
    # model = Sequential()
    # model.add(Masking(mask_value=0., input_shape=(subframe_length, num_features)))
    # model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    # model.add(tf.keras.layers.LSTM(64))
    # model.add(Dense(1, activation='linear'))
    # model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    # model.summary()

    # Train the model
    # Use the first 80% of the data for training and the last 20% for validation
    split_index = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_index]
    train_labels = labels[:split_index]
    val_sequences = sequences[split_index:]
    val_labels = labels[split_index:]
    model.fit(train_sequences, train_labels, epochs=num_epochs, batch_size=2, validation_data=(val_sequences, val_labels))
    model.save("models/cnn_model.keras")


if __name__ == '__main__':
    main()