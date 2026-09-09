import numpy as np
import tensorflow as tf
import random
import pandas as pd

OBSERVATION_SPACE = (210, 160, 6)
ACTION_SPACE = 6

last_init = tf.random_uniform_initializer(minval=0, maxval=0.000003)
input_x = tf.keras.Input(shape=OBSERVATION_SPACE)

x = tf.keras.layers.Conv2D(64, 3, activation="linear", padding="same")(input_x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, 3, activation="linear", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, 1, activation="linear", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation = "relu")(x)
x = tf.keras.layers.Dense(64, activation = "relu")(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
Actor = tf.keras.Model(inputs=input_x, outputs=x)




state_input = tf.keras.Input(shape=(OBSERVATION_SPACE))
action_input = tf.keras.Input(shape=(1))
xa = tf.keras.layers.Dense(32, activation = "relu")(action_input)
xa = tf.keras.layers.Dense(32, activation = "relu")(xa)

x = tf.keras.layers.Conv2D(64, 3, activation="linear", padding="same")(state_input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, 3, activation="linear", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, 1, activation="linear", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Concatenate()([x, xa])

x = tf.keras.layers.Dense(64, activation = "relu")(x)
x = tf.keras.layers.Dense(64, activation = "relu")(x)
x = tf.keras.layers.Dense(1, activation = "linear")(x)
Critic = tf.keras.Model(inputs=[state_input, action_input], outputs=x)

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critics_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

Actor.summary(), Critic.summary()