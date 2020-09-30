'''
common block
'''
import tensorflow as tf
slim = tf.contrib.slim


def conv(inputs, out_channels, kernel_size=3, down_sample=False):
    if down_sample:
        padding = 'SAME'
        stride = 1
    else:
        padding = 'VALID'
        stride = 2
    net = slim.conv2d(inputs, out_channels, kernel_size, stride=stride, padding=padding)
    return net


def csp_block(inputs, in_channels, res_block_sum, double_channels=False):
    out_channels = in_channels if double_channels else 2 * in_channels

    net = conv(inputs, in_channels * 2, down_sample=True)
    route = conv(net, out_channels, kernel_size=1)
    net = conv(net, out_channels, kernel_size=1)

    # res block
    for _ in range(res_block_sum):
        net = res_block(net, in_channels, double_channels=False)

    net = conv(net, out_channels, kernel_size=1)
    net = tf.concat([net, route], axis=-1)
    net = conv(net, in_channels * 2, kernel_size=1)
    return net


def res_block(inputs, in_channels, double_channels=False):
    out_channels = in_channels if double_channels else 2 * in_channels
    route = inputs
    net = conv(inputs, in_channels, kernel_size=1)
    net = conv(net, out_channels, kernel_size=3)
    net = route + net
    return net


def spp_block(inputs):
    net_5  = tf.nn.max_pool(inputs, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    net_9  = tf.nn.max_pool(inputs, ksize=[1, 9, 9, 1], strides=[1, 1, 1, 1], padding='SAME')
    net_13 = tf.nn.max_pool(inputs, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.concat([net_13, net_9, net_5, inputs], axis=-1)
    return net


# CBL * 5
def cbl_block(inputs, out_channels):
    net = conv(inputs, out_channels, kernel_size=1)
    net = conv(net, out_channels * 2)
    net = conv(net, out_channels, kernel_size=1)
    net = conv(net, out_channels * 2)
    net = conv(net, out_channels, kernel_size=1)
    return net


# CBL + upsample
def upsample_block(inputs, out_channels):
    net = conv(inputs, out_channels, kernel_size=1)

    # NHWC
    shape = tf.shape(net)
    new_h, new_w = shape[1] * 2, shape[2] * 2
    net = tf.compat.v1.image.resize_nearest_neighbor(net, (new_h, new_w))
    return net
