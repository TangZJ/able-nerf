
import numpy as np
import os 
from pdb import set_trace as st
dir = ""


psnr_file = os.path.join(dir,"psnrs.txt")

psnr = np.loadtxt(psnr_file, delimiter=' ')
psnr_mean = psnr.mean()



print("psnr mean is ", psnr_mean)

ssim_file = os.path.join(dir, "ssim.txt")

ssim = np.loadtxt(ssim_file, delimiter=' ')
ssim_mean = ssim.mean()
print("ssim is ", ssim_mean)
