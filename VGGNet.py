#-*- coding: UTF-8 -*-
"""
Author: lanbing510
Environment: Keras2.0.5，Python2.7
Model: VGGNet-19
"""

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
from KerasLayers.Custom_layers import LRN2D


# Global Constants
NB_CLASS=1000
LEARNING_RATE=0.01
MOMENTUM=0.9
ALPHA=0.0001
BETA=0.75
GAMMA=0.1
DROPOUT=0.5
WEIGHT_DECAY=0.0005
LRN2D_NORM=True
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'


def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY):
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None
    
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
            
    if lrn2d_norm:
        x=LRN2D(alpha=ALPHA,beta=BETA)(x)

    return x


def create_model():
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,224,224)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(224,224,3)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3
    else:
        raise Exception('Invalid Dim Ordering: '+str(DIM_ORDERING))
    
    # Convolution Net Layer 1~2
    x=conv2D_lrn2d(img_input,64,(3,3),1,padding='same',lrn2d_norm=False)
    x=conv2D_lrn2d(x,64,(3,3),1,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(2,2),strides=2,padding='valid',data_format=DATA_FORMAT)(x)
    
    # Convolution Net Layer 3~4
    x=conv2D_lrn2d(x,128,(3,3),1,padding='same',lrn2d_norm=False)
    x=conv2D_lrn2d(x,128,(3,3),1,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(2,2),strides=2,padding='valid',data_format=DATA_FORMAT)(x)

    # Convolution Net Layer 5~8
    x=conv2D_lrn2d(x,256,(3,3),1,padding='same',lrn2d_norm=False)
    x=conv2D_lrn2d(x,256,(3,3),1,padding='same',lrn2d_norm=False)    
    x=conv2D_lrn2d(x,256,(3,3),1,padding='same',lrn2d_norm=False)
    x=conv2D_lrn2d(x,256,(3,3),1,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(2,2),strides=2,padding='valid',data_format=DATA_FORMAT)(x)

    # Convolution Net Layer 9~12
    x=conv2D_lrn2d(x,512,(3,3),1,padding='same',lrn2d_norm=False)
    x=conv2D_lrn2d(x,512,(3,3),1,padding='same',lrn2d_norm=False)    
    x=conv2D_lrn2d(x,512,(3,3),1,padding='same',lrn2d_norm=False)
    x=conv2D_lrn2d(x,512,(3,3),1,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(2,2),strides=2,padding='valid',data_format=DATA_FORMAT)(x)

    # Convolution Net Layer 13~16
    x=conv2D_lrn2d(x,512,(3,3),1,padding='same',lrn2d_norm=False)
    x=conv2D_lrn2d(x,512,(3,3),1,padding='same',lrn2d_norm=False)    
    x=conv2D_lrn2d(x,512,(3,3),1,padding='same',lrn2d_norm=False)
    x=conv2D_lrn2d(x,512,(3,3),1,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(2,2),strides=2,padding='valid',data_format=DATA_FORMAT)(x)
    
    
    # Convolution Net Layer 17
    x=Flatten()(x)
    x=Dense(4096,activation='relu')(x)
    x=Dropout(DROPOUT)(x)

    # Convolution Net Layer 18
    x=Dense(4096,activation='relu')(x)
    x=Dropout(DROPOUT)(x)
    
    # Convolution Net Layer 19
    x=Dense(output_dim=NB_CLASS,activation='softmax')(x)
    
    return x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT


def check_print():
    # Create the Model
    x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT=create_model()

    # Create a Keras Model
    model=Model(input=img_input,output=[x])
    model.summary()

    # Save a PNG of the Model Build
    plot_model(model,to_file='VGGNet.png')
    
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    print 'Model Compiled'


if __name__=='__main__':
    check_print() 
