import os
import sys
import time
import numpy as np

def benchmark_raster():
    N = 160000
    H, W = 1080, 1920
    np.random.seed(42)
    
    centers = np.random.uniform([100, 100], [W - 100, H - 100], size=(N, 2)).astype(np.float32)
    depths = np.random.uniform(1.0, 10.0, size=N).astype(np.float32)
    colors = np.random.uniform(0.1, 0.9, size=(N, 3)).astype(np.float32)
    radii = np.random.randint(1, 5, size=N).astype(np.int32)
    opacities = np.full(N, 0.95, dtype=np.float32)
    
    # Sort front to back
    order = np.argsort(depths)
    centers = centers[order]
    colors = colors[order]
    radii = radii[order]
    opacities = opacities[order]
    
    t0 = time.time()
    out_img = np.zeros((H, W, 3), dtype=np.float32)
    trans = np.ones((H, W), dtype=np.float32)
    
    # Fast direct rasterization
    u_int = np.clip(np.round(centers[:, 0]).astype(np.int32), 0, W - 1)
    v_int = np.clip(np.round(centers[:, 1]).astype(np.int32), 0, H - 1)
    
    for i in range(N):
        r = radii[i]
        u = u_int[i]
        v = v_int[i]
        if r <= 1:
            if trans[v, u] > 1e-4:
                a = opacities[i]
                w = a * trans[v, u]
                out_img[v, u] += colors[i] * w
                trans[v, u] *= (1.0 - a)
        else:
            u0 = max(0, u - r)
            u1 = min(W, u + r + 1)
            v0 = max(0, v - r)
            v1 = min(H, v + r + 1)
            sub_trans = trans[v0:v1, u0:u1]
            if np.max(sub_trans) > 1e-4:
                # Local gaussian kernel
                du = np.arange(u0 - u, u1 - u, dtype=np.float32)
                dv = np.arange(v0 - v, v1 - v, dtype=np.float32)
                DU, DV = np.meshgrid(du, dv)
                dist_sq = (DU*DU + DV*DV) / (r * r)
                mask = dist_sq <= 1.0
                alpha = np.zeros_like(dist_sq)
                alpha[mask] = opacities[i] * np.exp(-1.5 * dist_sq[mask])
                
                weight = alpha * sub_trans
                out_img[v0:v1, u0:u1, 0] += colors[i, 0] * weight
                out_img[v0:v1, u0:u1, 1] += colors[i, 1] * weight
                out_img[v0:v1, u0:u1, 2] += colors[i, 2] * weight
                trans[v0:v1, u0:u1] *= (1.0 - alpha)
                
    elapsed = time.time() - t0
    print(f"Direct rasterization of {N:,} Gaussians on 1080p: {elapsed:.3f}s")

if __name__ == "__main__":
    benchmark_raster()
