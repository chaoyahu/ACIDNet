import glob
from tqdm import tqdm
from PIL import Image
import imquality.brisque as brisque
from acidnet.losses.niqe_utils import calculate_niqe
import argparse
import numpy as np
import skimage.color
import torch
import warnings
from pathlib import Path

from acidnet.paths import OUTPUT_ROOT

# 过滤掉 numpy 的 RuntimeWarning，保持输出整洁（因为我们在 niqe_utils 里已经处理了）
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================= 修复 skimage 的 Monkey Patch =================
_original_rgb2gray = skimage.color.rgb2gray
def safe_rgb2gray(img):
    if img.ndim == 2:
        return img
    return _original_rgb2gray(img)
skimage.color.rgb2gray = safe_rgb2gray
# ==============================================================

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--DICM', action='store_true',help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true',  help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true',help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true',default='true', help='output VV dataset')
eval_parser.add_argument('--pred-dir', type=str, default=None, help='prediction image directory or glob')
eval_parser.add_argument('--limit', type=int, default=None, help='optional image limit for smoke tests')
ep = eval_parser.parse_args()

def metrics(im_dir, limit=None):
    avg_niqe = 0
    avg_brisque = 0
    n = 0
    
    files = sorted(glob.glob(im_dir))
    if limit is not None:
        files = files[:limit]
    if not files:
        print(f"No files found in {im_dir}")
        return 0, 0
        
    for item in tqdm(files):
        try:
            # 1. 读取图片 (PIL 默认为 RGB)
            pil_img = Image.open(item).convert('RGB')
            im_rgb = np.array(pil_img)
            
            # 2. 检查图片尺寸
            # NIQE 要求至少 96x96 (默认 block size)
            h, w, _ = im_rgb.shape
            if h < 96 or w < 96:
                # print(f"Skipping small image: {item} ({h}x{w})")
                continue

            # 3. 计算 BRISQUE
            # 传入 RGB (H, W, 3), 范围 [0, 255]
            try:
                score_brisque = brisque.score(im_rgb)
            except Exception:
                # 兼容性处理
                score_brisque = brisque.score(pil_img)
            
            # 4. 计算 NIQE
            # ！！！关键点：NIQE Utils 内部期望 BGR 顺序！！！
            # 将 RGB 转为 BGR
            im_bgr = im_rgb[:, :, ::-1] 
            
            # calculate_niqe 内部会处理 BGR -> Y Channel 的转换
            score_niqe = calculate_niqe(im_bgr, crop_border=0)
            
            # 5. 结果校验与累加
            # 如果任何一个指标算出来是 NaN (例如纯黑图)，则跳过该图
            if np.isnan(score_niqe) or np.isinf(score_niqe):
                # print(f"Invalid NIQE for {item}")
                continue
                
            if np.isnan(score_brisque) or np.isinf(score_brisque):
                # print(f"Invalid BRISQUE for {item}")
                continue

            avg_brisque += score_brisque
            avg_niqe += score_niqe
            n += 1

            # 移除 torch.cuda.empty_cache()，因为这里的 NIQE/BRISQUE 主要在 CPU 上跑，
            # 频繁清理显存会极大地拖慢速度。

        except Exception as e:
            print(f"Error processing {item}: {e}")
            continue
    
    if n == 0:
        print("No valid images processed.")
        return 0, 0

    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque

if __name__ == '__main__':

    im_dir = ''
    if ep.pred_dir:
        pred_path = Path(ep.pred_dir)
        im_dir = str(pred_path if any(ch in ep.pred_dir for ch in "*?[]") else pred_path / "*")
    elif ep.DICM:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "DICM" / "*")
    elif ep.LIME:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "LIME" / "*")
    elif ep.MEF:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "MEF" / "*")
    elif ep.NPE:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "NPE" / "*")
    elif ep.VV:
        im_dir = str(OUTPUT_ROOT / "benchmark" / "VV" / "*")
    
    if im_dir:
        print(f"Evaluating dataset: {im_dir}")
        avg_niqe, avg_brisque = metrics(im_dir, ep.limit)
        print(f"Processed {im_dir} images.")
        print(f"Avg NIQE: {avg_niqe}")
        print(f"Avg BRISQUE: {avg_brisque}")
    else:
        print("Please specify a dataset (e.g., --VV)")
