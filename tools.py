from keras import backend as K
import tensorflow as tf
import os, time, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam,SGD
from skimage.measure import compare_psnr, compare_ssim
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,Multiply,Add,Concatenate
from keras import regularizers
from keras.utils import plot_model
from keras import initializers
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from pywt import dwt2,idwt2 
import pywt
import scipy.io as sio      


def load_train_data(train_data):
    
    print('loading train data...')   
    data = np.load(train_data)
    print('Size of train data: ({}, {}, {})'.format(data.shape[0],data.shape[1],data.shape[2]))
    
    return data

def Degrade(image):
    beta =  0.2*np.random.rand(1)
    image.astype('float32')    

    G_col =  np.random.normal(0, beta, image.shape[1])
    G_noise = np.tile(G_col,(image.shape[0],1))
    G_noise = np.reshape(G_noise,image.shape)
               
    image_G = image + G_noise
    return image_G, beta[0]
    
           
def train_datagen(y_, batch_size=128):
    # y_ is the tensor of clean patches
     indices = list(range(y_.shape[0]))
     steps_per_epoch = len(indices)//batch_size -1 
     j = 0
     while(True):
        np.random.shuffle(indices)    # shuffle
        ge_batch_y = []
        ge_batch_x = []
        for i in range(batch_size):
            sample = y_[indices[j*batch_size+i]]
            LLY,(LHY,HLY,HHY) = pywt.dwt2(sample, 'haar')
            Y = np.stack((LLY,LHY,HLY,HHY),axis=2)
            sample_O,_=  Degrade(sample)  # input image = clean image + noise
            LLX,(LHX,HLX,HHX) = pywt.dwt2(sample_O, 'haar')
            X = np.stack((LLX,LHX,HLX,HHX),axis=2)
            ge_batch_y.append(Y)
            ge_batch_x.append(X)
        if j == steps_per_epoch:
                j = 0
                np.random.shuffle(indices)
        else:
                j += 1
        yield np.array(ge_batch_x), {"residual": np.array(ge_batch_y),"res": np.array(ge_batch_y)}
        
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
  
    
def directional_tv(y_true, y_pred):
    P = 2
    acc = 0 
    b=0
    for l in range(0,P+1):           
                a = K.mean(K.square(y_pred[0,:-1,:,0] -  y_pred[0,1:,:,0]))  
                b = K.mean(K.square(y_pred[0,:-1,:,2] -  y_pred[0,1:,:,2]))                           
                acc = acc + a + b
    loss = acc/P
    return loss
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 1.0    
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_true - y_pred)))) 
def PSNR_LL(y_true, y_pred):
    max_pixel = 1.0    
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_true[:,:,:,0] - y_pred[:,:,:,0])))) 
def PSNR_HL(y_true, y_pred):
    max_pixel = 1.0    
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_true[:,:,:,1] - y_pred[:,:,:,1])))) 
def PSNR_LH(y_true, y_pred):
    max_pixel = 1.0    
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_true[:,:,:,2] - y_pred[:,:,:,2])))) 
def PSNR_HH(y_true, y_pred):
    max_pixel = 1.0    
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_true[:,:,:,3] - y_pred[:,:,:,3])))) 
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#