import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import platform
from pathlib import Path

from ACIDNet.paths import DATASET_ROOT, OUTPUT_ROOT



def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr

def metrics(im_dir, label_dir, use_GT_mean):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    files = sorted(glob.glob(im_dir))
    if not files:
        raise FileNotFoundError(f"No prediction images matched: {im_dir}")

    label_dir = str(label_dir)
    if label_dir and not label_dir.endswith(os.sep):
        label_dir += os.sep

    for item in tqdm(files):
        n += 1
        
        im1 = Image.open(item).convert('RGB') 
        
        os_name = platform.system()
        if os_name.lower() == 'windows':
            name = item.split('\\')[-1]
        elif os_name.lower() == 'linux':
            name = item.split('/')[-1]
        else:
            name = item.split('/')[-1]
            
        # im2 = Image.open(label_dir + name).convert('RGB')
        try:
            # 1. Try the filename directly (for example 1.jpg or 1.png).
            im2 = Image.open(label_dir + name).convert('RGB')
        except FileNotFoundError:
            try:
                # 2. Retry with an uppercase .JPG extension.
                im2 = Image.open(label_dir + name.replace('.jpg', '.JPG')).convert('RGB')
            except FileNotFoundError:
                # 3. Retry with an uppercase .PNG extension.
                im2 = Image.open(label_dir + name.replace('.png', '.PNG')).convert('RGB')
        
        (h, w) = im2.size
        im1 = im1.resize((h, w))  
        im1 = np.array(im1) 
        im2 = np.array(im2)
        
        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
            im1 = np.clip(im1 * (mean_target/mean_restored), 0, 255)
        
        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)
        ex_p0 = lpips.im2tensor(im1).cuda()
        ex_ref = lpips.im2tensor(im2).cuda()
        

        score_lpips = loss_fn.forward(ex_ref, ex_p0)
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips.item()
        torch.cuda.empty_cache()
    

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    return avg_psnr, avg_ssim, avg_lpips


if __name__ == '__main__':
    
    mea_parser = argparse.ArgumentParser(description='Measure')
    mea_parser.add_argument('--use_GT_mean', action='store_true' , help='Use the mean of GT to rectify the output of the model')
    mea_parser.add_argument('--lol', action='store_true' , help='measure lolv1 dataset')
    mea_parser.add_argument('--lol_v2_real', action='store_true',help='measure lol_v2_real dataset')
    mea_parser.add_argument('--lol_v2_syn', action='store_true',default='True', help='measure lol_v2_syn dataset')
    mea_parser.add_argument('--SICE_grad', action='store_true',  help='measure SICE_grad dataset')
    mea_parser.add_argument('--SICE_mix', action='store_true',help='measure SICE_mix dataset')
    mea_parser.add_argument('--fivek', action='store_true', help='measure fivek dataset')
    mea_parser.add_argument('--pred-dir', type=str, default=None, help='prediction image directory or glob')
    mea_parser.add_argument('--gt-dir', type=str, default=None, help='ground-truth image directory')
    mea = mea_parser.parse_args()

    if mea.pred_dir and mea.gt_dir:
        pred_path = Path(mea.pred_dir)
        im_dir = str(pred_path if any(ch in mea.pred_dir for ch in "*?[]") else pred_path / "*")
        label_dir = mea.gt_dir
    elif mea.lol:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "LOLv1" / "*.png")
        label_dir = DATASET_ROOT / "LOLdataset/eval15/high"
    elif mea.lol_v2_real:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "LOLv2_real" / "*.png")
        label_dir = DATASET_ROOT / "LOLv2/Real_captured/Test/Normal"
    elif mea.lol_v2_syn:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "LOLv2_syn" / "*.png")
        label_dir = DATASET_ROOT / "LOLv2/Synthetic/Test/Normal"
    elif mea.SICE_grad:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "SICE_grad" / "*.png")
        label_dir = DATASET_ROOT / "SICE/SICE_Reshape"
    elif mea.SICE_mix:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "SICE_mix" / "*.png")
        label_dir = DATASET_ROOT / "SICE/SICE_Reshape"
    elif mea.fivek:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "FiveK" / "*.jpg")
        label_dir = DATASET_ROOT / "FiveK/test/target"
    else:
        raise SystemExit("Specify a dataset flag or --pred-dir and --gt-dir.")

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, mea.use_GT_mean)
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
