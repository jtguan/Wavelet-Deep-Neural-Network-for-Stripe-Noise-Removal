import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import argparse


def normalize(data):
    return data/255.




def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    tlist=os.listdir(data_path)
    tlist.sort()
    Ffile = []
    for i in tlist:
        datapath=os.listdir(os.path.join(data_path,i))
        datapath.sort()

        for j in datapath:
                path=os.path.join(data_path,i,j)
                Ffile.append(path)
    files = Ffile
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i]) 
        Img = np.expand_dims(img[:,:,0].copy(), 0)
        
        Img = np.float32(normalize(Img))
        
        patches = Im2Patch(Img, win=patch_size, stride=stride)
        print("file: %s  # samples: %d" % (files[i], patches.shape[3]))
        for n in range(patches.shape[3]):
            data = patches[:,:,:,n].copy()
            h5f.create_dataset(str(train_num), data=data)
            train_num += 1
            
    h5f.close()
    
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join('val', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
          

if __name__ == "__main__":
    patch_size = 64
    stride = 64
    data_path = "IRdataset"
    prepare_data(data_path, patch_size, stride)

