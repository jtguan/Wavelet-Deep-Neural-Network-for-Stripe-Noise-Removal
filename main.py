# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 19:08:40 2018

@author: jtguan@stu.xidian.edu.cn
"""


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
from tools import load_train_data, Degrade, train_datagen, directional_tv, tf_log10, PSNR, PSNR_LL, PSNR_HL, PSNR_LH, PSNR_HH
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
L2 =None
init = 'he_normal'
    
def SNRDWNN():
    
    inpt = Input(shape=(None,None,4))
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same' ,kernel_initializer=init,name='Conv-1')(inpt)
    x = Activation('relu')(x)
    for i in range(8):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same' ,kernel_initializer=init)(x)
        x = Activation('relu')(x)
    residual = Conv2D(filters=4, kernel_size=(3,3), strides=(1,1), padding='same' ,kernel_initializer=init, name = 'residual')(x)
    res = Add(name = 'res')([inpt,residual])
    model = Model(inputs=inpt, 
                  outputs=[res,residual],
                  name = 'DWSRN'
                  )
    
    return model


#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#


      

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#

batch_size = 128
train_data = '../data/npy_data/Train_64.npy'
checkpoint_file = 'weights'
TRAIN = 0
TEST  = 1
realFrame = 0
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def adam_step_decay(epoch):
    
    if epoch<= 20:
        lr = 1e-3
    elif epoch > 20 and epoch <= 35:
        lr = 1e-3 * 0.5
    elif epoch > 35 and epoch <= 45:
        lr = 1e-4
    else:
        lr = 1e-4 *0.5
        
if TRAIN:      
    data = load_train_data(train_data)
    data = data.reshape((data.shape[0],data.shape[1],data.shape[2]))
    data = data.astype('float32')/255.0
        

    model =  SNRDWNN()
    # model selection
    with open('./'+checkpoint_file+'/model_summary.txt','w') as f:
        model.summary(print_fn=lambda x: f.write(x+'\n'))


    #plot_model(model, to_file='./'+'./'+checkpoint_file+'/model.png')
    opt = Adam(decay =1e-7)
    
    losses = {
	"res": "mse",
	"residual": directional_tv,
     }
    lossWeights = {"res": 0.95, "residual": 0.05}
    

    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=[PSNR_LL,PSNR_HL,PSNR_LH,PSNR_HH])
    
    # use call back functions
    filepath='./'+checkpoint_file+'/weights-{epoch:02d}-{res_PSNR_LL:.4f}-{res_PSNR_HL:.4f}-{res_PSNR_LH:.4f}-{res_PSNR_HH:.4f}-{res_loss:.4f}.hdf5'
    ckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, period=1)
    lr = LearningRateScheduler(adam_step_decay)
    TensorBoard = keras.callbacks.TensorBoard(log_dir='./logs')
    # train 

    history = model.fit_generator(train_datagen(data, batch_size=batch_size),
								  steps_per_epoch=len(data)//batch_size, 
								  epochs=50,verbose=1, callbacks=[ckpt, lr, TensorBoard],initial_epoch=0)
    


#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
if TEST:
    WEIGHT_PATH = './'+checkpoint_file+'/weight.hdf5'
    if not realFrame:
        print('Test on Simulation SEQ or IMG!')
        save_dir = './sim_res'
        test_dir = './Set12/'        
        #----------------------------------------------------------------------#
        def Addnoise(image,beta= 0.15):
            image.astype('float32') 
            np.random.seed(0)
            G_col =  np.random.normal(0, beta, image.shape[1])
            G_noise = np.tile(G_col,(image.shape[0],1))
            G_noise = np.reshape(G_noise,image.shape)
            
            image_G = image + G_noise
            return image_G
        #----------------------------------------------------------------------#
        

        model =  SNRDWNN()
        model.load_weights(WEIGHT_PATH)
        print('Start to test on {}'.format(test_dir))
        out_dir = save_dir + '/' + test_dir.split('/')[-1] + '/'
        if not os.path.exists(out_dir):
                os.mkdir(out_dir)
                
        name = []
        GainList = []
        OnoiseList = []
        psnr = []
        ssim = []
        file_list = os.listdir(test_dir)
        cnt=0
        for file in file_list:
            # read image
            cnt = cnt + 1
            IMG = Image.open(test_dir + file)
            img_clean = np.array(IMG, dtype='float32') / 255.0
            img_test  = Addnoise(img_clean,beta=0.1).astype('float32')
            # predict
            x_test = np.expand_dims(img_test,axis=0)
            x_test = np.expand_dims(x_test,axis=3)                

            # Descreat Wavelet Transform
            t1 = time.time()
            LLY,(LHY,HLY,HHY) = pywt.dwt2(img_test, 'haar')
            Y = np.stack((LLY,LHY,HLY,HHY),axis=2)
            # predict
            x_test = np.expand_dims(Y,axis=0)
            y_pred,noise = model.predict(x_test)
            # calculate numeric metrics

            coeffs_pred = y_pred[0,:,:,0],(LHY,y_pred[0,:,:,2],HHY)

            img_out = pywt.idwt2(coeffs_pred, 'haar')
            t2 = time.time()
            print("time: %f" % (t2-t1))
            img_out = np.clip(img_out, 0, 1)
            psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
            ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test), compare_ssim(img_clean, img_out)
            psnr.append(psnr_denoised)
            ssim.append(ssim_denoised)
            # save images
            filename = str(cnt)
            name.append(filename)
            img_test = Image.fromarray(np.clip((img_test*255),0,255).astype('uint8'))
            img_test.save(out_dir + filename+'_dsrn_psnr{:.2f}.png'.format(psnr_noise))
            img_out = Image.fromarray((img_out*255).astype('uint8')) 
            img_out.save(out_dir + filename+'dwsrn__psnr{:.2f}.png'.format(psnr_denoised))
            
            print(filename)
            
        
        psnr_avg = sum(psnr)/len(psnr)
        ssim_avg = sum(ssim)/len(ssim)
        name.append('Average')
        psnr.append(psnr_avg)
        ssim.append(ssim_avg)
        print('Average PSNR = {0:.4f}, SSIM = {1:.4f}'.format(psnr_avg, ssim_avg))
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
if(realFrame):
	#----------------------------------------------------------------------#
	print("Test on Real Frame !")
	save_dir = './realFrame'
	multi_GPU = 0
	#----------------------------------------------------------------------#
	test_dir = './IR_Set/'
	

	model =  SNRDWNN()
	model.load_weights(WEIGHT_PATH)
	print('Start to test on {}'.format(test_dir))
	out_dir = save_dir + '/' + test_dir.split('/')[-1] + '/'
	if not os.path.exists(out_dir):
			os.mkdir(out_dir)
			
	name = []

	file_list = os.listdir(test_dir)
	for file in file_list:
		# read image
		img_clean = np.array(Image.open(test_dir + file), dtype='float32') / 255.0
		img_test = img_clean.astype('float32')
		if(len(img_test.shape)>2):
			img_test = img_test[:,:,0]
		# predict
		x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1) 
		LLY,(LHY,HLY,HHY) = pywt.dwt2(img_test, 'haar')
		Y = np.stack((LLY,LHY,HLY,HHY),axis=2)
		# predict
		x_test = np.expand_dims(Y,axis=0)
		y_pred,noise = model.predict(x_test)
		# calculate numeric metrics
		pred = np.stack((y_pred[0,:,:,0],y_pred[0,:,:,1],y_pred[0,:,:,2],y_pred[0,:,:,3]),axis=2)
		coeffs_pred = y_pred[0,:,:,0],(y_pred[0,:,:,1],y_pred[0,:,:,2],y_pred[0,:,:,3])
		img_out = pywt.idwt2(coeffs_pred, 'haar')
		# calculate numeric metrics
		img_out = np.clip(img_out, 0, 1)
		filename = file    # get the name of image file
		name.append(filename)
		img_out = Image.fromarray((img_out*255).astype('uint8')) 
		img_out.save(out_dir + filename)
		
		print('save'+out_dir + filename)
	
	print('Test Over')
                