
import keras.backend as K
import tensorflow as tf


def huber_loss(target, prediction, clip_delta=1.0):
    # Implementation from: https://stackoverflow.com/a/48791563/1207773
    error = target - prediction
    cond = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    # Implementation from: https://stackoverflow.com/a/48791563/1207773
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def pseudo_huber_loss(target, prediction, delta=1.0):
    # Pseudo-Huber loss from: https://en.wikipedia.org/wiki/Huber_loss
    error = prediction - target
    return K.mean(K.sqrt(1 + K.square(error / delta)) - 1, axis=-1)
