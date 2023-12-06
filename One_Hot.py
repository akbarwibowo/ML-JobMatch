import tensorflow as tf
import numpy as np


def to_one_hot(data):
    num_data = len(data)

    data_mapping = {label: i for i, label in enumerate(data)}

    data_encoded = np.array([data_mapping[label] for label in data], dtype=np.int32)

    data_one_hot = tf.keras.utils.to_categorical(data_encoded, num_data)

    return data_one_hot
