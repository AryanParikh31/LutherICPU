import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from benchmark.evaluator import calculate_psnr, calculate_ssim, generate_side_by_side_comparison

pred = Image.open('output/truck_proof_render.png')
gt = Image.open(r'F:\tandt_db\tandt\truck\images\000001.jpg').resize(pred.size, Image.Resampling.BILINEAR)

p_arr = np.array(pred)
g_arr = np.array(gt)

psnr = calculate_psnr(p_arr, g_arr)
ssim = calculate_ssim(p_arr, g_arr)

print(f"EVALUATION RESULT -> PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

generate_side_by_side_comparison(
    gt_img_path=r'F:\tandt_db\tandt\truck\images\000001.jpg',
    pred_img=pred,
    output_path='output/truck_comparison.png',
    scene_name='truck',
    psnr_val=psnr,
    ssim_val=ssim
)
