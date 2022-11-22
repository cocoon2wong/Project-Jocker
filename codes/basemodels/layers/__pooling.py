"""
@Author: Conghao Wong
@Date: 2022-11-21 10:15:13
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-22 09:25:45
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


class _BasePooling2D(tf.keras.layers.Layer):
    """
    The base pooling layer that supports both CPU and GPU.
    """

    pool_function: type[tf.keras.layers.MaxPooling2D] = None

    def __init__(self, pool_size=(2, 2), strides=None,
                 padding: str = 'valid',
                 data_format: str = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.gpu = is_gpu()
        self.data_format = data_format
        self.pool_layer = self.pool_function(pool_size, strides,
                                             padding, data_format, **kwargs)

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        """
        Run the 2D pooling operation.

        :param inputs: The input tensor, shape = `(batch, channels, a, b)`
        """
        # Pool layer with 'channels_first' runs only on gpus
        if (not self.gpu) and (self.data_format == 'channels_first'):
            # Reshape the input to (batch, a, b, channels)
            i_reshape = tf.transpose(inputs, [0, 2, 3, 1])
            pooled = self.pool_layer(i_reshape)
            return tf.transpose(pooled, [0, 3, 1, 2])
        else:
            return self.pool_layer(inputs)


class MaxPooling2D(_BasePooling2D):

    pool_function = tf.keras.layers.MaxPooling2D

    def __init__(self, pool_size=(2, 2), strides=None,
                 padding: str = 'valid',
                 data_format: str = None,
                 *args, **kwargs):

        super().__init__(pool_size, strides, padding,
                         data_format, *args, **kwargs)


def is_gpu():
    gpu_devices = tf.config.list_physical_devices('GPU')
    if len(gpu_devices):
        return True
    else:
        return False
