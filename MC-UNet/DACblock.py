from keras.layers import concatenate, Conv2D,Activation
from keras.layers import Lambda

# nonlinearity = functools.partial(relu)

# def conv1(x, filter):
#     return tf.nn.atrous_conv2d(value=x, filters=filter, rate=1, padding='VALID')

def dilate1x1(input_feature,filter,kernel_size,rate,padding):
    return Conv2D(filters = filter,kernel_size = (kernel_size,kernel_size),activation='relu', dilation_rate=(1, 1), padding=padding)(input_feature)

def dilate3x3(input_feature,filter,kernel_size, rate,padding):
    return Conv2D(filters = filter,kernel_size = (kernel_size,kernel_size),activation='relu', dilation_rate=rate, padding=padding)(input_feature)


# def dilate(input_feature,channel,rate,padding):
#     return tf.nn.atrous_conv2d(value=input_feature, filters=channel, rate=rate, padding=padding)

def DACblock(input_feature, channel):

        dilate1 = dilate3x3(input_feature, channel, 3, 1, 'SAME') #dilate1(x)
        dilate1_out = dilate1

        dilate2 = dilate3x3(input_feature, channel, 3, 3, 'SAME') #dilate2(x)
        dilate2_out = dilate1x1(dilate2, channel, 1, 1,'SAME')

        dilate3 = dilate3x3(input_feature, channel, 3, 1, 'SAME') #dilate3(x)
        dilate3 = dilate3x3(dilate3, channel, 3, 3, 'SAME')
        dilate3_out = dilate1x1(dilate3, channel, 1, 1, 'SAME')

        dilate4 = dilate3x3(input_feature, channel, 3, 1, 'SAME')  # dilate3(x)
        dilate4 = dilate3x3(dilate4, channel, 3, 3, 'SAME')
        dilate4 = dilate3x3(dilate4, channel, 3, 5, 'SAME')
        dilate4_out = dilate1x1(dilate4, channel, 1, 1, 'SAME')

        my_concat = Lambda(lambda x: concatenate([x[0], x[1]], axis=-1))


        # out = input_feature + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        # # out = Add()([input_feature, dilate1_out,dilate2_out,dilate3_out,dilate4_out])
        # # out = Concatenate(axis=3)([input_feature, dilate1_out,dilate2_out,dilate3_out,dilate4_out])
        # # out = concatenate([input_feature, dilate1_out])
        # # out = concatenate([out, dilate2_out])
        # # out = concatenate([out, dilate3_out])
        # # out = concatenate([out, dilate4_out])
        out = my_concat([input_feature, dilate1_out])
        out = my_concat([out, dilate2_out])
        out = my_concat([out, dilate3_out])
        out = my_concat([out, dilate4_out])
        return out

def RDACblock(input_feature, channel):

        dilate1 = dilate3x3(input_feature, channel, 3, 1, 'SAME') #dilate1(x)
        dilate1_out = dilate1

        dilate2 = dilate3x3(input_feature, channel, 3, 3, 'SAME') #dilate2(x)
        dilate2_out = dilate1x1(dilate2, channel, 1, 1,'SAME')

        dilate3 = dilate3x3(input_feature, channel, 3, 1, 'SAME') #dilate3(x)
        dilate3 = dilate3x3(dilate3, channel, 3, 3, 'SAME')
        dilate3_out = dilate1x1(dilate3, channel, 1, 1, 'SAME')

        my_concat = Lambda(lambda x: concatenate([x[0], x[1]], axis=-1))

        out = my_concat([input_feature, dilate1_out])
        out = my_concat([out, dilate2_out])
        out = my_concat([out, dilate3_out])
        return out