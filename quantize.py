import tensorflow as tf

# tf.RegisterGradient('QuantizeGradient')
from tensorflow_core.python import ops


def clamp(input_tensor, num_bits=8):
    input_tensor = tf.clip_by_value(input_tensor, clip_value_min=0, clip_value_max=2 ** (num_bits) - 1)
    input_tensor = ((input_tensor * 2 ** (num_bits)).round()) / 2 ** (num_bits)
    return input_tensor


def to_fixed_point(input_tensor, ibits, fbits):
    input_tensor_i = tf.sign(input_tensor) * clamp(tf.floor(tf.abs(input_tensor)), ibits)
    input_tensor_f = tf.sign(input_tensor) * clamp(tf.abs(input_tensor) - tf.floor(tf.abs(input_tensor)), fbits)
    return input_tensor_i + input_tensor_f


@tf.RegisterGradient('fixed_point_grad')
# https://uoguelph-mlrg.github.io/tensorflow_gradients/
# https://stackoverflow.com/questions/43256517/how-to-register-a-custom-gradient-for-a-operation-composed-of-tf-operations
# https://github.com/keras-team/keras/issues/8526
def _qunatize_grad(ops, grad):
    return grad


@tf.custom_gradient
def quantize(x):
    y = to_fixed_point(x, ibits=4, fbits=4)

    def _fixed_point_grad(dy):
        grad = dy
        return grad

    return y, _fixed_point_grad


class QuantizeLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(QuantizeLayer, self).__init__()

    def call(self, x):
        return quantize(x)


class Conv2dQuant(tf.keras.layers.Conv2D):
    def __init__(self):
        
