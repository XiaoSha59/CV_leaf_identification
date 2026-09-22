from typing import List, Tuple
import numpy as np
from scipy.ndimage import map_coordinates


def get_uniform_pattern_groups(P: int = 8) -> Tuple[dict, list, int]:
    pattern_map = {}
    for code in range(2**P):
        bits = [(code >> i) & 1 for i in range(P)]
        transitions = sum(bits[i] != bits[(i + 1) % P] for i in range(P))
        n_ones = sum(bits)
        
        if transitions <= 2:
            if n_ones == 0:
                pattern_map[code] = ("all_0", 0)
            elif n_ones == P:
                pattern_map[code] = ("all_1", 0)
            else:
                for r in range(P):
                    rotated_code = ((code >> r) | (code << (P - r))) & ((1 << P) - 1)
                    canonical_code = (1 << n_ones) - 1
                    if rotated_code == canonical_code:
                        pattern_map[code] = (n_ones, r)
                        break
                else:
                    pattern_map[code] = (n_ones, 0)
        else:
            pattern_map[code] = ("non_uniform", 0)
            
    group_n_list = list(range(1, P))
    n_features = len(group_n_list) * (P // 2 + 1) + 3
    return pattern_map, group_n_list, n_features


PATTERN_MAP_P8, GROUP_N_P8, N_HF_P8 = get_uniform_pattern_groups(P=8)


def sample_circular_neighbors(
    image: np.ndarray,
    P: int = 8,
    radius: float = 1.0,
) -> np.ndarray:
    H, W = image.shape
    y_coords, x_coords = np.indices((H, W), dtype=np.float32)
    
    neighbors = np.empty((P, H, W), dtype=np.float32)
    for p in range(P):
        angle = 2.0 * np.pi * p / P
        x_p = x_coords + radius * np.cos(angle)
        y_p = y_coords - radius * np.sin(angle)
        
        np.clip(x_p, 0, W - 1, out=x_p)
        np.clip(y_p, 0, H - 1, out=y_p)
        
        neighbors[p] = map_coordinates(
            image, [y_p.ravel(), x_p.ravel()], order=1, mode="nearest"
        ).reshape(H, W)
        
    return neighbors


def compute_clbp_maps(
    image_gray: np.ndarray,
    P: int = 8,
    radius: float = 1.0,
    mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    img_f = image_gray.astype(np.float32)
    H, W = img_f.shape
    neighbors = sample_circular_neighbors(img_f, P=P, radius=radius)
    diffs = neighbors - img_f[np.newaxis, :, :]
    
    sign_bits = (diffs >= 0).astype(np.uint8)
    sign_code = np.zeros((H, W), dtype=np.uint16)
    for p in range(P):
        sign_code |= (sign_bits[p] << p)
        
    abs_diffs = np.abs(diffs)
    if mask is not None and np.sum(mask > 0) > 0:
        valid_pixels = (mask > 0)
        t_p = np.mean(abs_diffs[:, valid_pixels], axis=1, keepdims=True)[:, :, np.newaxis]
    else:
        t_p = np.mean(abs_diffs, axis=(1, 2), keepdims=True)
        
    mag_bits = (abs_diffs >= t_p).astype(np.uint8)
    mag_code = np.zeros((H, W), dtype=np.uint16)
    for p in range(P):
        mag_code |= (mag_bits[p] << p)
        
    return sign_code, mag_code


def compute_lbp_hf_features(
    code_map: np.ndarray,
    region_mask: np.ndarray | None = None,
    P: int = 8,
) -> np.ndarray:
    if region_mask is not None and np.sum(region_mask > 0) > 0:
        valid_codes = code_map[region_mask > 0]
    else:
        valid_codes = code_map.ravel()
        
    h_rot = {n: np.zeros(P, dtype=np.float64) for n in GROUP_N_P8}
    count_all_0 = 0
    count_all_1 = 0
    count_non_uniform = 0
    
    codes, counts = np.unique(valid_codes, return_counts=True)
    for c, cnt in zip(codes, counts):
        group, r = PATTERN_MAP_P8.get(int(c), ("non_uniform", 0))
        if group == "all_0":
            count_all_0 += cnt
        elif group == "all_1":
            count_all_1 += cnt
        elif group == "non_uniform":
            count_non_uniform += cnt
        else:
            h_rot[group][r] += cnt
            
    fourier_features = []
    n_freqs = P // 2 + 1
    
    for n in GROUP_N_P8:
        hist_1d = h_rot[n]
        dft_coeffs = np.fft.rfft(hist_1d, n=P)
        magnitudes = np.abs(dft_coeffs)
        fourier_features.extend(magnitudes[:n_freqs])
        
    fourier_features.append(float(count_all_0))
    fourier_features.append(float(count_all_1))
    fourier_features.append(float(count_non_uniform))
    
    feat_vector = np.array(fourier_features, dtype=np.float32)
    norm = np.sum(feat_vector)
    if norm > 0:
        feat_vector /= norm
        
    return feat_vector


def extract_lbp_hf_sm(
    image_gray: np.ndarray,
    region_mask: np.ndarray | None = None,
    P: int = 8,
    radius: float = 1.0,
) -> np.ndarray:
    sign_code, mag_code = compute_clbp_maps(
        image_gray, P=P, radius=radius, mask=region_mask
    )
    feat_s = compute_lbp_hf_features(sign_code, region_mask=region_mask, P=P)
    feat_m = compute_lbp_hf_features(mag_code, region_mask=region_mask, P=P)
    return np.concatenate([feat_s, feat_m]).astype(np.float32)


if __name__ == "__main__":
    test_patch = (np.random.rand(64, 64) * 255).astype(np.uint8)
    f1 = extract_lbp_hf_sm(test_patch, radius=1.0)
    test_patch_rot = np.rot90(test_patch)
    f2 = extract_lbp_hf_sm(test_patch_rot, radius=1.0)
    diff = np.mean(np.abs(f1 - f2))
    print(f"LBP-HF-S-M feature dimension: {f1.shape[0]}D (Expected 76D)")
    print(f"Mean feature difference under 90-degree rotation: {diff:.6f}")
