# SNRWDNN-PyTorch

This is a PyTorch implementation of the  paper [Wavelet Deep Neural Network for Stripe Noise Removal]. 
****
This code was written with PyTorch 1.8.1 and CUDA10.1. 
****

## How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)


### 1. Train SNRWDNN (SNRWDNN with known noise level)
```
python train.py \
  --preprocess True \
  --data-path *your_datapath* \
  --batchSize 128 \
  --epochs 50 \
  --log-interval 10 \
  --lr 1e-3 \
  --outf *your_log_file* \
```
**NOTE**
* *preprocess* run prepare_data to get .h5 file or not.
* *outf* is path of your train log file.


### 2. val SNRWDNN (stripe noise)
```
python val.py \
  --Beta 0.1 \
  --model-dir *your_path* \
  --outf *your_log_file* \
```
**NOTE**
* *Beta* is stripe noise beta.
* *model-dir* path of your train log files.
* *outf* is path of your val log files.


### 3. Test
```
python test.py \
  --log-path *your_log_file*\
  --filename *your_test_image*\
```




