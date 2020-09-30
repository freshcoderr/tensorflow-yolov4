'''
yolov4 network
'''
import tensorflow as tf
from model.common import conv, csp_block, spp_block, cbl_block, upsample_block
slim = tf.contrib.slim


class YOLOV4(object):

    def __init__(self, class_num, weight_decay, batch_norm_params):
        self.class_num = class_num
        self.weight_decay = weight_decay
        self.batch_norm_params = batch_norm_params
        pass

    # backbone: Darknet53
    def backbone(self, inputs, batch_norm_params, weight_decay):
        # =============== Mish activation ===============
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            activation_fn=Mish, # TODO: Mish func
                            weights_refularizer=slim.l2_regularizer(weight_decay)):
            # CBM
            net = conv(inputs, 32)
            # CSP1
            net = csp_block(net, 32, res_block_sum=1, double_channels=True)
            # CSP2
            net = csp_block(net, 64, res_block_sum=2, double_channels=False)
            # CSP8, feature_map_76
            net = csp_block(net, 128, res_block_sum=8, double_channels=False)
            route_76 = net
            # CSP8, feature_map_38
            net = csp_block(net, 256, res_block_sum=8, double_channels=False)
            route_38 = net
            # CSP4
            net = csp_block(net, 512, res_block_sum=4, double_channels=False)

        # =============== LeakyRelu activation ===============
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            activation_fn=lambda input: tf.nn.leaky_relu(input),
                            weights_refularizer=slim.l2_regularizer(weight_decay)):
            # CBL * 3
            net = conv(net, 512, kernel_size=1)
            net = conv(net, 1024)
            net = conv(net, 512, kernel_size=1)
            # SPP, shape:[19, 19, 1024]
            net = spp_block(net)
            # CBL * 3, feature_map_19
            net = conv(net, 512, kernel_size=1)
            net = conv(net, 1024)
            route_19 = conv(net, 512, kernel_size=1)
        return route_19, route_38, route_76


    def forward(self, inputs):
        with tf.variable_scope('backbone_Darknet53'):
            # [19, 19, 512], [38, 38, 512], [78, 78, 256]
            route_19, route_38, route_76 = self.backbone(inputs, self.batch_norm_params, self.weight_decay)
        with tf.variable_scope('yolov4_head'):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params,
                                activation_fn=lambda x: tf.nn.leaky_relu(x),
                                weights_refularizer=slim.l2_regularizer(self.weight_decay)):
                # feature_map_76
                fpn_19 = upsample_block(route_19, 256)
                route_38 = conv(route_38, 256)
                route_38 = tf.concat([route_38, fpn_19], axis=-1)
                route_38 = cbl_block(route_38, 256)
                fpn_38 = upsample_block(route_38, 128)
                route_76 = conv(route_76, 128)
                route_76 = tf.concat([route_76, fpn_38], axis=-1)
                route_76 = cbl_block(route_76, 128)
                pan_76 = route_76
                feature_76 = conv(route_76, 256)

                # feature_map_38
                pan_76 = conv(pan_76, 256, down_sample=True)
                route_38 = tf.concat([route_38, pan_76], axis=-1)
                route_38 = cbl_block(route_38, 256)
                pan_38 = route_38
                feature_38 = conv(route_38, 512)

                # feature_map_19
                pan_38 = conv(pan_38, 512, down_sample=True)
                route_19 = tf.concat([route_19, pan_38], axis=-1)
                route_19 = cbl_block(route_19, 512)
                feature_19 = (route_19, 1024)
            feature_19 = slim.conv2d(feature_19, 3*(4+1+self.class_num), 1, stride=1,
                                     normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer())
            feature_38 = slim.conv2d(feature_38, 3 * (4 + 1 + self.class_num), 1, stride=1,
                                     normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer())
            feature_76 = slim.conv2d(feature_76, 3 * (4 + 1 + self.class_num), 1, stride=1,
                                     normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer())
            return feature_19, feature_38, feature_76

    def featureMap_decode(self):
        pass

    def __iou_loss(self):
        pass

    def __giou_loss(self):
        pass

    def __ciou_loss(self):
        pass

    def compute_loss(self):
        pass
