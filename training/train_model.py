import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model # type: ignore
from keras.layers import Masking, Conv1D, GlobalAveragePooling1D, Dense # type: ignore

subframe_length = 30

def parse_dataset(file_path):
    video_names = []
    labels = []
    sequences = []
    with open(file_path, "r") as f:
        videos = f.read().strip().split("\n\n\n")
        for video in videos:
            lines = video.split("\n")
            video_name = lines[0]
            slalom_points = eval(lines[1])
            frames = [[component for keypoint in eval(lines[i]) for component in keypoint] for i in range(2, len(lines))]
            # Split frames into subframes of 30 frames each
            subframes = [frames[i-subframe_length:i] for i in range(subframe_length, len(frames))]
            video_names += [video_name] * (len(frames) - subframe_length)
            labels += [slalom_points] * (len(frames) - subframe_length)
            sequences += subframes
    sequences = np.array(sequences)
    labels = np.array(labels)
    return video_names, labels, sequences

def compute_avg(x):
    return tf.reduce_mean(x[:, :-1, :], axis=1)

def main():
    num_epochs = 7
    video_names, labels, sequences = parse_dataset("dataset.txt")
    num_frames = subframe_length
    num_features = 34

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(num_frames, num_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    model.summary()

    model.fit(sequences, labels, epochs=16, batch_size=2)
    model.save("models/model.keras")

if __name__ == '__main__':
    main()