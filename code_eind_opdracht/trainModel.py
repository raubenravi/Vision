import tensorflow as tf
from tensorflow_examples.profiling.resnet_model import layers
import numpy as np

def train_model(patches, labels):
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model = tf.keras.models.Sequential([
        #bron chatgpt
        tf.keras.layers.Input(shape=(32, 32, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'],
                  run_eagerly=True)

    labels = np.reshape(labels, (-1, 1))
    #labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    print(patches.shape)
    print(labels.shape)
    print(labels[0])
    model.fit(patches, labels, epochs=20)
    model.save('saved_model/my_model')