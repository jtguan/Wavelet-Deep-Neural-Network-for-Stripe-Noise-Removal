import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from snrwdnn_model import SNRWDNN
from utils import batch_PSNR
import cv2
import pywt


parser = argparse.ArgumentParser(description="snrwdnn")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--Beta", type=float, default=0.1, help='stride noise beta')
parser.add_argument("--val_noiseG", type=float, default=0, help='noise level used on validation set')
parser.add_argument("--model_dir", type=str, default="train_stripe_haar_pth", help='path of train log files')
parser.add_argument("--outf", type=str, default="val_haar_pth", help='path of val log files')
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

for epoch in range(opt.epochs):
    psnr_val = 0 
    s = "epoch_" + str(epoch) + "_snrwdnn_train.pth"
    model = SNRWDNN(channels=4, num_of_layers=8)
    #Move to GPU
    device_ids = [0]
    #model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.model_dir, s)))
    
    
    file = []
    val_list = os.listdir('val')
    val_list.sort()
    for i in val_list:
        datapath = os.path.join('val', i)
        file.append(datapath)
    for k in range(len(file)):
        image = cv2.imread(file[k])
        image = modcrop(image)
        img = np.expand_dims(image[:,:,0], 0)
        img = np.float32(img/255.)
        img = torch.Tensor(img)
        img_val = torch.unsqueeze(img, 0)
        
#        np.random.seed(1)
#        noise_G = np.random.normal(0, opt.val_noiseG/255., img_val.shape)
        np.random.seed(1)
        G_col =  np.random.normal(0, opt.Beta, img_val.shape[3])
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
        imgn, img_val = Variable(imgn.cuda()), Variable(img_val.cuda())
        imgn = imgn.float()
        out_haar = model(imgn)
    
        z = out_haar.data.cpu().numpy()      
        coeffs = z[0,0,:,:],(z[0,1,:,:],z[0,2,:,:],z[0,3,:,:])

        out = pywt.idwt2(coeffs, 'haar')
        
        m = np.expand_dims(out, 0)
        img_out = np.expand_dims(m,0)
        img_out = torch.Tensor(img_out)
        out_val = torch.clamp(img_out, 0., 1.)   
    
        psnr_val += batch_PSNR(out_val, img_val, 1.)
            
    psnr_val /= len(file)
    if psnr_val>psnr_max:
        psnr_max = psnr_val
    else:
        psnr_max = psnr_max
    print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
    # save model
    p = "epoch_" + str(epoch) + "_PSNR_" + str(psnr_val) + "_val.pth"
    torch.save(model.state_dict(), os.path.join(opt.outf, p))
    model.eval()
print(psnr_max)

