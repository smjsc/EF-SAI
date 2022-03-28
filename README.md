# EF-SAI

Paper | [EF-SAI Dataset](https://github.com/smjsc/EF-SAI#Dataset) | [Pre-trained Models](https://1drv.ms/u/s!AhglJgt1Cr16pXzr5N4RlkvYgzDe?e=Ar21Gw) | BibTex

**Synthetic Aperture Imaging with Events and Frames**<br>

_Wei Liao, Xiang Zhang, Lei Yu, Shijie Lin, Wen Yang, Ning Qiao_<br>
In CVPR'2022

## Dataset
The EF-SAI dataset is used for model training and evaluation. It can be download from [Onedrive](https://1drv.ms/u/s!AhglJgt1Cr16pXLLwDzp7rnbGMdS?e=hI6okp)
## Requirements:
     python 3.6
     pytorch 1.7.1
     torchvision 0.8.2
     numpy, opencv, timm, matlpotlib, prefetch_generator
## Code Implementation
### Installation
1. Clone this repo
2. Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org)
3. Install required python packages
### Preparations:
1. Download validation data from [Onedrive](https://1drv.ms/u/s!AhglJgt1Cr16pXpFl2DBfDe98pkh?e=AMonte) to ./EF-SAI-main/Val_data folder
2. Download pre-trained model from [Onedrive](https://1drv.ms/u/s!AhglJgt1Cr16pXzr5N4RlkvYgzDe?e=Ar21Gw) to ./EF-SAI-main/Pretrained folder
### Testing
1. Change the parent directory to ./EF-SAI-main
2. Run testing script
```
python test.py 
```
3. The results will stored in the Results folder.
