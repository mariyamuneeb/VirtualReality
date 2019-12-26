
import tensorflow as tf


# tf.RegisterGradient('QuantizeGradient')

def clamp(input_tensor, num_bits = 8):
    input_tensor = tf.clip_by_value(input_tensor, clip_value_min=0, clip_value_max =2 ** (num_bits) - 1)
    input_tensor = ((input_tensor * 2 ** (num_bits)).round()) / 2 ** (num_bits)
    return input_tensor

def to_fixed_point(input_tensor,ibits,fbits):
    input_tensor_i = tf.sign(input_tensor)*clamp(tf.floor(tf.abs(input_tensor)),ibits)
    input_tensor_f = tf.sign(input_tensor)*clamp(tf.abs(input_tensor)-tf.floor(tf.abs(input_tensor)),fbits)
    return input_tensor_i + input_tensor_f

@tf.RegisterGradient('fixed_point_grad')
#https://uoguelph-mlrg.github.io/tensorflow_gradients/
#https://stackoverflow.com/questions/43256517/how-to-register-a-custom-gradient-for-a-operation-composed-of-tf-operations
# https://github.com/keras-team/keras/issues/8526
def _qunatize_grad(ops,grad):
    return grad

def quantize(input_tensor):
    G = tf.compat.v1.get_default_graph()
    with G.over_ride_gradient_map






def quantize_grad():
    None

def quantize(x,n):
    max_ =