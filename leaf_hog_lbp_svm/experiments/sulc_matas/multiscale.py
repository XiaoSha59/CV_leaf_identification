import sys
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.sulc_matas.lbp_hf import compute_clbp_maps, compute_lbp_hf_features
from experiments.sulc_matas.segmentation import segment_leaf_mask, split_interior_border


def compute_scale_parameters(
    n_conc: int = 3,
    c: int = 6,
    P: int = 8,
) -> tuple[list[float], list[float]]:
    total_scales = n_conc + c - 1
    radii = []
    sigmas = []
    
    current_r = 1.0
    for _ in range(total_scales):
        radii.append(current_r)
        sigma = current_r * np.sin(np.pi / P) / 1.96
        sigmas.append(sigma)
        current_r *= np.sqrt(2.0)
        
    return radii, sigmas


def extract_all_regions_multiscale(
    image_gray: np.ndarray,
    n_conc: int = 3,
    c: int = 6,
    P: int = 8,
    target_width: int = 400,
) -> Dict[str, List[np.ndarray]]:
    # Optional speedup resize while preserving aspect ratio and leaf texture
    H, W = image_gray.shape
    if W > target_width:
        scale_factor = target_width / W
        new_H = int(round(H * scale_factor))
        img_proc = cv2.resize(image_gray, (target_width, new_H), interpolation=cv2.INTER_AREA)
    else:
        img_proc = image_gray

    leaf_mask = segment_leaf_mask(img_proc)
    radii, sigmas = compute_scale_parameters(n_conc=n_conc, c=c, P=P)
    total_scales = len(radii)

    scale_descs = {"all": [], "interior": [], "border": []}

    for i in range(total_scales):
        r_i = radii[i]
        sigma_i = sigmas[i]

        if sigma_i > 0.6:
            filtered_img = gaussian_filter(img_proc.astype(np.float32), sigma=sigma_i)
        else:
            filtered_img = img_proc.astype(np.float32)

        interior_mask, border_mask = split_interior_border(leaf_mask, radius=r_i)

        # Compute CLBP code maps once per scale
        sign_code, mag_code = compute_clbp_maps(filtered_img, P=P, radius=r_i, mask=leaf_mask)

        # Extract histograms for each region from the same code maps
        for reg_name, reg_mask in [("all", leaf_mask), ("interior", interior_mask), ("border", border_mask)]:
            f_s = compute_lbp_hf_features(sign_code, region_mask=reg_mask, P=P)
            f_m = compute_lbp_hf_features(mag_code, region_mask=reg_mask, P=P)
            desc_76 = np.concatenate([f_s, f_m]).astype(np.float32)
            scale_descs[reg_name].append(desc_76)

    # Concatenate c adjacent scales for each of the n_conc multi-scale descriptors
    final_multiscale = {"all": [], "interior": [], "border": []}
    for reg_name in ["all", "interior", "border"]:
        for j in range(n_conc):
            selected = scale_descs[reg_name][j : j + c]
            fused = np.concatenate(selected).astype(np.float32)
            norm = np.sum(fused)
            if norm > 0:
                fused /= norm
            final_multiscale[reg_name].append(fused)

    return final_multiscale


if __name__ == "__main__":
    from time import perf_counter
    test_img = cv2.imread("data/raw/Flavia/1001.jpg", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        t0 = perf_counter()
        res = extract_all_regions_multiscale(test_img)
        t_elapsed = perf_counter() - t0
        print(f"Extracted all 3 regions in: {t_elapsed:.3f}s")
        print(f"Descriptors per region: {len(res['interior'])} x {res['interior'][0].shape[0]}D")
