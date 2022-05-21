
from keras.layers import concatenate, Conv2D,Activation,MaxPooling2D
from keras.layers import Lambda
from keras.activations import relu
import cv2
import tensorflow as tf
import numpy as np


def my_upsampling(x, img_w, img_h, method=0):
    """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
    return tf.image.resize_images(x, (img_w, img_h), 0)


def conv1x1(input_feature,kernel_size,padding):
    return Conv2D(filters = 1,kernel_size = (kernel_size,kernel_size),activation='relu', padding=padding)(input_feature)


# def bilinear_interpolation(src, new_size):
#
#         dst_w, dst_h = new_size  # 目标图像宽高
#         src_h, src_w = src.shape[1].value,src.shape[2].value  # 源图像宽高
#         if src_h == dst_h and src_w == dst_w:
#             return src.copy()
#         dst_w, dst_h = dst_w.value, dst_h.value
#
#         scale_x = float(src_w) / dst_w  # x缩放比例
#         scale_y = float(src_h) / dst_h  # y缩放比例
#
#
#         # 遍历目标图像，插值
#         dst = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
#         dst = tf.convert_to_tensor(dst, tf.float32, name='dst')
#
#         for n in range(3):  # 对channel循环
#             for dst_y in range(dst_h):  # 对height循环
#                 for dst_x in range(dst_w):  # 对width循环
#                     # 目标在源上的坐标
#                     src_x = (dst_x + 0.5) * scale_x - 0.5
#                     src_y = (dst_y + 0.5) * scale_y - 0.5
#                     # 计算在源图上四个近邻点的位置
#                     src_x_0 = int(np.floor(src_x))
#                     src_y_0 = int(np.floor(src_y))
#                     src_x_1 = min(src_x_0 + 1, src_w - 1)
#                     src_y_1 = min(src_y_0 + 1, src_h - 1)
#
#
#                     s1 = src[src_y_0, src_x_0, n]
#                     s2 = src[src_y_0, src_x_1, n]
#                     s3 = src[src_y_1, src_x_0, n]
#                     s4 = src[src_y_1, src_x_1, n]
#
#
#
#                     # 双线性插值
#                     value0 = (src_x_1 - src_x) * tf.to_float(s1, name='ToFloat') + (src_x - src_x_0) * tf.to_float(s2, name='ToFloat')
#                     value1 = (src_x_1 - src_x) * tf.to_float(s3, name='ToFloat') + (src_x - src_x_0) * tf.to_float(s4, name='ToFloat')
#
#                     sess = tf.Session()
#                     sess.run(tf.global_variables_initializer())
#                     # print("out1=", type(s1))
#                     # 转化为numpy数组
#                     img_numpy = s1.eval(session=sess)
#                     # print("out2=", type(img_numpy))
#
#                     # print(value1.numpy())
#                     # print(tf.to_float(s1, name='ToFloat'))
#
#
#                     dst[dst_y, dst_x, n] = (src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1
#
#
#
#         return dst




def SPPblock(input_feature):

        my_concat = Lambda(lambda x: concatenate([x[0], x[1]], axis=-1))


        pool1 = MaxPooling2D((2, 2),strides=2)(input_feature)
        pool2 = MaxPooling2D((3, 3),strides=3)(input_feature)
        pool3 = MaxPooling2D((5, 5),strides=5)(input_feature)
        pool4 = MaxPooling2D((6, 6),strides=6)(input_feature)

        layer1 = conv1x1(pool1, 1, 'VALID')
        layer2 = conv1x1(pool2, 1, 'VALID')
        layer3 = conv1x1(pool3, 1, 'VALID')
        layer4 = conv1x1(pool4, 1, 'VALID')


        layer11 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]})\
            (layer1)
        layer22 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]})\
            (layer2)
        layer33 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]})(
            layer3)
        layer44 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]})(
            layer4)

        # layer1 = bilinear_interpolation(layer1, (input_feature.shape[1],input_feature.shape[2]))
        # layer2 = bilinear_interpolation(layer2, (input_feature.shape[1],input_feature.shape[2]))
        # layer3 = bilinear_interpolation(layer3, (input_feature.shape[1],input_feature.shape[2]))
        # layer4 = bilinear_interpolation(layer4, (input_feature.shape[1],input_feature.shape[2]))


        out = my_concat([input_feature, layer11])
        out = my_concat([out, layer22])
        out = my_concat([out, layer33])
        out = my_concat([out, layer44])

        return out


def RSPPblock(input_feature):
    my_concat = Lambda(lambda x: concatenate([x[0], x[1]], axis=-1))

    pool1 = MaxPooling2D((2, 2), strides=2)(input_feature)
    pool2 = MaxPooling2D((3, 3), strides=3)(input_feature)
    pool3 = MaxPooling2D((5, 5), strides=5)(input_feature)
    pool4 = MaxPooling2D((6, 6), strides=6)(input_feature)
    pool5 = MaxPooling2D((7, 7), strides=7)(input_feature)
    pool6 = MaxPooling2D((9, 9), strides=9)(input_feature)

    layer1 = conv1x1(pool1, 1, 'VALID')
    layer2 = conv1x1(pool2, 1, 'VALID')
    layer3 = conv1x1(pool3, 1, 'VALID')
    layer4 = conv1x1(pool4, 1, 'VALID')
    layer5 = conv1x1(pool5, 1, 'VALID')
    layer6 = conv1x1(pool6, 1, 'VALID')

    layer11 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]}) \
        (layer1)
    layer22 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]}) \
        (layer2)
    layer33 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]})(
        layer3)
    layer44 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]})(
        layer4)

    layer55 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]})(
        layer5)
    layer66 = Lambda(my_upsampling, arguments={'img_w': input_feature.shape[1], 'img_h': input_feature.shape[2]})(
        layer6)

    # layer1 = bilinear_interpolation(layer1, (input_feature.shape[1],input_feature.shape[2]))
    # layer2 = bilinear_interpolation(layer2, (input_feature.shape[1],input_feature.shape[2]))
    # layer3 = bilinear_interpolation(layer3, (input_feature.shape[1],input_feature.shape[2]))
    # layer4 = bilinear_interpolation(layer4, (input_feature.shape[1],input_feature.shape[2]))

    out = my_concat([input_feature, layer11])
    out = my_concat([out, layer22])
    out = my_concat([out, layer33])
    out = my_concat([out, layer44])
    out = my_concat([out, layer55])
    out = my_concat([out, layer66])
    return out