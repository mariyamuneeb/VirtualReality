# import tensorflow as tf
# from tf.keras.layers import Input
# from tf.keras.layers import Conv2D
# from tf.keras.models import Model
#
# import tensorflow
# # tf.RegisterGradient('QuantizeGradient')
# from tensorflow_core.python import ops
#
#
# def clamp(input_tensor, num_bits=8):
#     input_tensor = tf.clip_by_value(input_tensor, clip_value_min=0, clip_value_max=2 ** (num_bits) - 1)
#     input_tensor = ((input_tensor * 2 ** (num_bits)).round()) / 2 ** (num_bits)
#     return input_tensor
#
#
# def to_fixed_point(input_tensor, ibits, fbits):
#     input_tensor_i = tf.sign(input_tensor) * clamp(tf.floor(tf.abs(input_tensor)), ibits)
#     input_tensor_f = tf.sign(input_tensor) * clamp(tf.abs(input_tensor) - tf.floor(tf.abs(input_tensor)), fbits)
#     return input_tensor_i + input_tensor_f
#
#
# @tf.RegisterGradient('fixed_point_grad')
# # https://uoguelph-mlrg.github.io/tensorflow_gradients/
# # https://stackoverflow.com/questions/43256517/how-to-register-a-custom-gradient-for-a-operation-composed-of-tf-operations
# # https://github.com/keras-team/keras/issues/8526
# def _qunatize_grad(ops, grad):
#     return grad
#
#
# @tf.custom_gradient
# def quantize(x):
#     y = to_fixed_point(x, ibits=4, fbits=4)
#
#     def _fixed_point_grad(dy):
#         grad = dy
#         return grad
#
#     return y, _fixed_point_grad
#
#
# class QuantizeLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(QuantizeLayer, self).__init__()
#
#     def call(self, x):
#         return quantize(x)
#
#
# class Conv2dQuant(tf.keras.layers.Conv2D):
#     def __init__(self, number_of_filters,
#                  filter_size,
#                  strides,
#                  dilation_rate=(1, 1),
#                  padding='same',
#                  kernel_initializer='he_normal'):
#         self.number_of_filter = number_of_filters
#         self.filter_size = filter_size
#         self.strides = strides
#         self.dilation_rate = dilation_rate
#         self.padding = padding
#         self.kernel_initializer = kernel_initializer
#         super(Conv2dQuant).__init__()
#
#     def call(self, inputs):
#         input_shape = K.shape(inputs)
#
#
#
# input = Input(shape=(1,4))
# out = Conv2D()
import matplotlib.pyplot as plt
loss_quant_unet_cross_eval = [3.01
,2.822
,2.89
,2.84
,2.91
,2.36
,2.47
,2.57
,2.996
,2.696
,2.579
,2.705
,2.819
,2.956
,2.772
,2.903
,2.809
,2.819
,2.992
,2.972
,3.105
,3.116
,3.252
,3.232
,3.355
,3.363
,3.356
]


