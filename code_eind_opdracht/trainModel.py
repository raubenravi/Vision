import tensorflow as tf
from tensorflow_examples.profiling.resnet_model import layers
import numpy as np


def train_model(patches, labels, patch_size=(32, 32), extra_layer_bool=True):
    if (extra_layer_bool):
        layers = 2
    else:
        layers = 1
    shape_of_patch = (patch_size[0], patch_size[1], layers)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model = tf.keras.models.Sequential([
        # bron chatgpt
        tf.keras.layers.Input(shape=shape_of_patch),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((1, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((1, 1)),
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
    # labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    # print(patches.shape)
    # print(labels.shape)
    # print(labels[0])
    model.fit(patches, labels, epochs=1)
    # model.save('saved_model/my_model')
    return model
