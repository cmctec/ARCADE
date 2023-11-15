from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, Softmax, Add, Dropout, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.layers import Concatenate as concatenate
import tensorflow as tf


def resblock(input_):
    input_ = BatchNormalization()(input_)
    conv_1 = Conv2D(input_.shape[3], (3, 3), activation = 'relu', padding='same')(input_)
    conv_2 = Conv2D(input_.shape[3], (3, 3), activation = 'relu', padding='same')(conv_1)
    batch_1 = BatchNormalization()(conv_2)
    add = Add()([input_,batch_1])

    conv_1 = Conv2D(input_.shape[3], (3, 3), activation = 'relu', padding='same')(add)
    conv_2 = Conv2D(input_.shape[3], (3, 3), activation = 'relu', padding='same')(conv_1)
    batch_1 = BatchNormalization()(conv_2)
    add = Add()([input_,batch_1])

    conv_1 = Conv2D(input_.shape[3], (3, 3), activation = 'relu', padding='same')(add)
    conv_2 = Conv2D(input_.shape[3], (3, 3), activation = 'relu', padding='same')(conv_1)
    batch_1 = BatchNormalization()(conv_2)
    
    add = Add()([input_,batch_1])
    batch = BatchNormalization()(add)
    return batch

def model():
    input_1 = Input(shape=(512, 512, 1))
    conv = Conv2D(16, (3, 3), padding='same')(input_1)
    
    res_1 = resblock(conv)
    conv_1 = Conv2D(32, (2, 2), strides = 2, padding='same')(res_1)
    
    res_2 = resblock(conv_1)
    conv_2 = Conv2D(64, (2, 2), strides = 2, padding='same')(res_2)
    
    res_3 = resblock(conv_2)
    conv_3 = Conv2D(128, (2, 2), strides = 2, padding='same')(res_3)
    
    res_4 = resblock(conv_3)
    conv_4 = Conv2D(256, (2, 2), strides = 2, padding='same')(res_4)
    
    res_7 = resblock(conv_4)
    res_8 = resblock(res_7)
    res_9 = resblock(res_8)
    
    trconv_3 = Conv2DTranspose(128, (2, 2), strides = 2, padding='same')(res_9)
    res_12 = resblock(trconv_3)
    add_3 = Add()([res_12, res_4])
    
    trconv_4 = Conv2DTranspose(64, (2, 2), strides = 2, padding='same')(add_3)
    res_13 = resblock(trconv_4)
    add_4 = Add()([res_13, res_3])
    
    trconv_5 = Conv2DTranspose(32, (2, 2), strides = 2, padding='same')(add_4)
    res_14 = resblock(trconv_5)
    add_5 = Add()([res_14, res_2])
    
    trconv_6 = Conv2DTranspose(16, (2, 2), strides = 2, padding='same')(add_5)
    res_15 = resblock(trconv_6)
    add_6 = Add()([res_15, res_1])
    
    conv_7 = Conv2D(16, (3, 3), activation = 'relu', padding='same')(add_6)
    batch_1 = BatchNormalization()(conv_7)
    conv_8 = Conv2D(16, (3, 3), activation = 'relu', padding='same')(batch_1)
    batch_2 = BatchNormalization()(conv_8)
    out1 = Conv2D(2, (1, 1), activation = 'softmax', padding='same', name='seg')(batch_2)
    
    model = keras.Model(inputs=[input_1], outputs=[out1], name= "model1") 
    
    return model