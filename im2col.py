
import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable

#import NeuralODE_MNIST_test

use_cuda = torch.cuda.is_available()


# Implement any ordinary differential equation initial value solver. For the sake of simplicity it'll be Euler's ODE initial value solver, however any explicit or implicit method will do.


src = torch.rand(10, 5, 9, 9)
kernel = torch.Tensor([[[[-0.1158,  0.0942, -0.0708],
          [-0.0048,  0.0526,  0.0757],
          [-0.0708, -0.1363, -0.0870]],

         [[-0.1139, -0.1128,  0.0702],
          [ 0.0631,  0.0857, -0.0244],
          [ 0.1197,  0.1481,  0.0765]],

         [[-0.0823, -0.0589, -0.0959],
          [ 0.0966,  0.0166,  0.1422],
          [-0.0167,  0.1335,  0.0729]],

         [[-0.0032, -0.0768,  0.0597],
          [ 0.0083, -0.0754,  0.0867],
          [-0.0228, -0.1440, -0.0832]],

         [[ 0.1352,  0.0615, -0.1005],
          [ 0.1163,  0.0049, -0.1384],
          [ 0.0440, -0.0468, -0.0542]]],
                       
        [[[-0.1158,  0.0942, -0.0708],
         [-0.0048,  0.0526,  0.0757],
         [-0.0708, -0.1363, -0.0870]],

        [[-0.1139, -0.1128,  0.0702],
         [ 0.0631,  0.0857, -0.0244],
         [ 0.1197,  0.1481,  0.0765]],

        [[-0.0823, -0.0589, -0.0959],
         [ 0.0966,  0.0166,  0.1422],
         [-0.0167,  0.1335,  0.0729]],

        [[-0.0032, -0.0768,  0.0597],
         [ 0.0083, -0.0754,  0.0867],
         [-0.0228, -0.1440, -0.0832]],

        [[ 0.1352,  0.0615, -0.1005],
         [ 0.1163,  0.0049, -0.1384],
         [ 0.0440, -0.0468, -0.0542]]],
        
        [[[-0.1158,  0.0942, -0.0708],
          [-0.0048,  0.0526,  0.0757],
          [-0.0708, -0.1363, -0.0870]],

         [[-0.1139, -0.1128,  0.0702],
          [ 0.0631,  0.0857, -0.0244],
          [ 0.1197,  0.1481,  0.0765]],

         [[-0.0823, -0.0589, -0.0959],
          [ 0.0966,  0.0166,  0.1422],
          [-0.0167,  0.1335,  0.0729]],

         [[-0.0032, -0.0768,  0.0597],
          [ 0.0083, -0.0754,  0.0867],
          [-0.0228, -0.1440, -0.0832]],

         [[ 0.1352,  0.0615, -0.1005],
          [ 0.1163,  0.0049, -0.1384],
          [ 0.0440, -0.0468, -0.0542]]]
         ])

scrN, srcChannel, intH, intW= src.shape
KoutChannel, KinChannel, kernel_H, kernel_W = kernel.shape
im2col_kernel = kernel.reshape(KoutChannel, -1)

outChannel, outH, outW =  KoutChannel, (intH - kernel_H + 1) , (intW - kernel_W + 1)
OutScrIm2Col = torch.zeros( [ kernel_H*kernel_W*KinChannel, outH*outW ] )

row_num, col_num = OutScrIm2Col.shape

ii, jj, cnt_row, cnt_col = 0, 0, 0, 0

# 卷积核的reshape准备 ：outchannel, k*k*inchannel
im2col_kernel = kernel.reshape(KoutChannel, -1)
# 输入的reshape准备 ：outH = (intH - k + 2*pading)/stride + 1 
outChannel, outH, outW =  KoutChannel, (intH - kernel_H + 1) , (intW - kernel_W + 1)
Out = torch.zeros((1, KoutChannel, outH*outW))
Out_bs = torch.zeros((KoutChannel, outH*outW))

for Outim2colCol_bs in range(0, src.shape[0]):
    i_id = -1
    cnt_col = -1
    cnr = 0
    
    for Outim2colCol_H in range(0, outH):
        i_id += 1
        j_id = -1
        cnt_row  = -1
        for Outim2colCol_W in range(0, outW):
            j_id += 1
            cnt_col += 1
            cnt_row = 0
            for c in range(0, srcChannel): # 取一次卷积的数据，放到一列
                for iii in range(0, kernel_H):
                    i_number = iii + i_id
                    for jjj in range(0, kernel_W):
                        j_number = jjj + j_id
                        OutScrIm2Col[cnt_row][cnt_col] = src[Outim2colCol_bs][c][i_number][j_number]
                        cnr +=1
                        cnt_row += 1
    if Outim2colCol_bs ==0: 
        Out_bs = torch.mm(im2col_kernel, OutScrIm2Col)
        Out = Out_bs.expand(1, KoutChannel, outH*outW)
    else:
        Out_bs = torch.mm(im2col_kernel, OutScrIm2Col)
        Out = torch.cat((Out, Out_bs.expand(1, KoutChannel, outH*outW)), dim = 0)

print(Out.shape) 
Out = Out.reshape(-1, outChannel, outH, outW)
print(Out.shape)    
