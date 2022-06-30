import cv2
import numpy as np
import torch
import pywt
import argparse

parser = argparse.ArgumentParser(description="snrwdnn test")
parser.add_argument("--log-path", type=str, default="", help='path of val log file')
parser.add_argument("--filename", type=str, default="", help='path of test image file')
opt = parser.parse_args()


def modcrop(image,scale = 2):
    if len(image.shape)==3:
        h,w,_ = image.shape
        h = h - np.mod(h,scale)
        w = w - np.mod(w,scale)
        image = image[0:h,0:w,:]
    else:
        h,w = image.shape
        h = h - np.mod(h,scale)
        w = w - np.mod(w,scale)
        image = image[0:h,0:w]
    return image

psnr_max = 0


from snrwdnn_model import SNRWDNN
model = SNRWDNN(channels=4, num_of_layers=8)
model.load_state_dict(torch.load(opt.log_path))

image = cv2.imread(opt.filename)
image = modcrop(image)
img = np.expand_dims(image[:,:,0], 0)
img = np.float32(img/255.)
img = torch.Tensor(img)
img_val = torch.unsqueeze(img, 0)

#np.random.seed(1)
#noise_G = np.random.normal(0, 0/255., img_val.shape)
np.random.seed(1)
G_col =  np.random.normal(0, 0.1, img_val.shape[3])
G_noise = np.tile(G_col,(img_val.shape[2],1))
G_noise = np.expand_dims(G_noise,0)
G_noise = np.expand_dims(G_noise,0)
    
imgn_val = img_val + G_noise 

b,c,h,w = imgn_val.size()
imgn_haar = torch.zeros(b,4,h//2,w//2)

LLY,(LHY,HLY,HHY) = pywt.dwt2(imgn_val[0,:,:,:], 'haar')
Y = np.concatenate((LLY,LHY,HLY,HHY),axis=0)
Y = np.expand_dims(Y,0)
        
imgn = torch.from_numpy(Y)
imgn = imgn.float()
out_haar = model(imgn)

z = out_haar.data.cpu().numpy()      
coeffs = z[0,0,:,:],(z[0,1,:,:],z[0,2,:,:],z[0,3,:,:])

out = pywt.idwt2(coeffs, 'haar')

out_val = np.expand_dims(out, 0)
out_val = np.transpose(out_val,(1,2,0))
out_val = out_val * 255
cv2.imwrite("denoise.png", out_val.astype("uint8"))

noise_img = imgn_val[0,:,:,:]
noise_img = noise_img.numpy()
noise_img = np.transpose(noise_img,(1,2,0))
noise_img = noise_img * 255
cv2.imwrite("noise.png", out_val.astype("uint8"))











