import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from snrwdnn_model import SNRWDNN
from generate_h5_dataset import prepare_data
from dataset import Dataset
from utils import batch_PSNR,weights_init_kaiming
import pywt


parser = argparse.ArgumentParser(description="snrwdnn")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--patch-size", type=int, default=64, help="size of image patch")
parser.add_argument("--stride", type=int, default=64, help="stride of patch")
parser.add_argument("--data-path", type=str, default="IRdataset", help='path of dataset images')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--log-interval", type=int, default=10, help="Number of batches to wait before log")
parser.add_argument("--milestone", type=int, default=20, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="train_stripe_haar_pth", help='path of log files')

opt = parser.parse_args()

if opt.preprocess:
    prepare_data(data_path=opt.data_path, patch_size=opt.patch_size, stride=opt.stride)
# Load dataset
print('Loading dataset ...\n')
dataset_train = Dataset(train=True)
dataset_val = Dataset(train=False)
loader_train = DataLoader(dataset=dataset_train, num_workers=20, batch_size=opt.batchSize, shuffle=True)
print("# of training samples: %d\n" % int(len(dataset_train)))
# Build model
model = SNRWDNN(channels=4, num_of_layers=8).cuda()
model.apply(weights_init_kaiming)
criterion = nn.MSELoss()
#Move to GPU
device_ids = [1]
#model = nn.DataParallel(net, device_ids=device_ids).cuda()
criterion.cuda()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
# training
noiseL_G=[0,15]
noiseB_S=[0.05,0.1]
step = 0
for epoch in range(opt.epochs):
    if epoch < opt.milestone:
        current_lr = 1e-3
    elif epoch > 20 and epoch <= 35:
        current_lr = 1e-3 * 0.5
    elif epoch > 35 and epoch <= 45:
        current_lr = 1e-4
    else:
        current_lr = 1e-4 *0.5
    # set learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    print('learning rate %f' % current_lr)
    # train
    for j, data in enumerate(loader_train):
        # training step
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        img_train = data
        
        # add gaussian noise 
        noise_G = torch.zeros(img_train.size())
        stdN_G = np.random.uniform(noiseL_G[0], noiseL_G[1], size=noise_G.size()[0])
        for n in range(noise_G.size()[0]):
            sizeN_G = noise_G[0,:,:,:].size()
            noise_G[n,:,:,:] = torch.FloatTensor(sizeN_G).normal_(mean=0, std=stdN_G[n]/255.)
            
        # add stride noise    
        noise_S = torch.zeros(img_train.size())
        beta = np.random.uniform(noiseB_S[0], noiseB_S[1], size=noise_S.size()[0])
        for m in range(noise_S.size()[0]):
            sizeN_S = noise_S[0,0,:,:].size()
            noise_col = np.random.normal(0, beta[m], sizeN_S[1])
            S_noise = np.tile(noise_col,(sizeN_S[0],1))
            S_noise = np.expand_dims(S_noise,0)
            S_noise = torch.from_numpy(S_noise) 
            noise_S[m,:,:,:] = S_noise
                
        # trans haar            
        imgn_train = img_train + noise_S 
        t,c,h,w = imgn_train.size()
        imgn = torch.zeros(t,4,h//2,w//2)
        orig = torch.zeros(t,4,h//2,w//2)
        for i in range(t):
            im = imgn_train[i,:,:,:]
            ori = img_train[i,:,:,:]
            LLY,(LHY,HLY,HHY) = pywt.dwt2(im, 'haar')
            lly,(lhy,hly,hhy) = pywt.dwt2(ori, 'haar')
            Y = np.concatenate((LLY,LHY,HLY,HHY),axis=0)
            y = np.concatenate((lly,lhy,hly,hhy),axis=0)
            t_Y = torch.from_numpy(Y)
            t_y = torch.from_numpy(y)
            imgn[i,:,:,:] = t_Y 
            orig[i,:,:,:] = t_y
        
                
        imgn, orig = Variable(imgn.cuda()), Variable(orig.cuda())
        noise_G = Variable(noise_G.cuda())    
        noise_S = Variable(noise_S.cuda())               
               
        out_train = model(imgn)
        loss = criterion(out_train, orig) 
        loss.backward()
        optimizer.step()
        step += 1
        if opt.log_interval and not (j+1)%opt.log_interval:
            print("[epoch %d][%d/%d] loss: %.4f" %
                 (epoch+1, j+1, len(loader_train), loss.item()))
    ## the end of each epoch
    s = "epoch_" + str(epoch) + "_snrwdnn_train.pth"
    torch.save(model.state_dict(), os.path.join(opt.outf, s))