import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

print(x_train.shape)
print(x_test.shape)
# This is going to be in numpy arrays
# you could convert it to tensor by
# x_train = tf.convert_to_tensor(x_train)
# But this is done internally by tf


# Sequential API - very convenient but not very flexible (One input One output)
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10)
    ]
)

# print(model.summary())

# Functional API - More flexible // But use only if you cant use sequential // Can handle multiple inp/outputs

inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation="relu", name="First_layer")(inputs)
x = layers.Dense(256, activation="relu", name="Second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Takes care of softmax layer
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)


# 1875/1875 - 4s - loss: 0.1315 - accuracy: 0.9646
# 313/313 - 0s - loss: 0.1449 - accuracy: 0.9636

# 1875/1875 - 5s - loss: 0.0331 - accuracy: 0.9891
# 313/313 - 0s - loss: 0.0859 - accuracy: 0.9750
