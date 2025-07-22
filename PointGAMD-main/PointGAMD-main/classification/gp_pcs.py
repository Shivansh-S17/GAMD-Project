import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def compute_surface_variation_gpu(xyz_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute surface variation (curvature-like feature) for each point in batched point clouds.

    Args:
        xyz_tensor (torch.Tensor): shape [B, 3, N], float32

    Returns:
        np.ndarray: shape [B, N], surface variation values
    """
    B, _, N = xyz_tensor.shape
    xyz = xyz_tensor.permute(0, 2, 1).cpu().numpy()  # [B, N, 3]
    variations = []

    for i in range(B):
        pts = xyz[i]
        neighbors = NearestNeighbors(n_neighbors=16).fit(pts)
        _, idx = neighbors.kneighbors(pts)

        curvatures = []
        for j in range(N):
            neighbor_pts = pts[idx[j]]
            centroid = np.mean(neighbor_pts, axis=0)
            cov = np.cov((neighbor_pts - centroid).T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)
            variation = eigvals[0] / (np.sum(eigvals) + 1e-6)
            curvatures.append(variation)

        variations.append(curvatures)

    return np.array(variations)  # shape [B, N]


def greedy_subset_of_data_gpu(xyz_np: np.ndarray, variation_np: np.ndarray, M: int,
                               k_init: int = 256, k_add: int = 256) -> np.ndarray:
    """
    Perform greedy point selection using GP-PCS logic on batched data.

    Args:
        xyz_np (np.ndarray): [B, N, 3] - input point clouds
        variation_np (np.ndarray): [B, N] - variation values
        M (int): number of points to sample
        k_init (int): number of top curvature points to initialize
        k_add (int): number of points to consider in each greedy addition

    Returns:
        np.ndarray: [B, M, 3] sampled coordinates
    """
    B, N, _ = xyz_np.shape
    all_samples = []

    for b in range(B):
        pts = xyz_np[b]
        var = variation_np[b]

        top_indices = np.argsort(-var)[:k_init]
        candidates = pts[top_indices]

        selected = [candidates[0]]
        while len(selected) < M:
            rest = np.array([pt for pt in candidates if not any(np.allclose(pt, s) for s in selected)])
            if rest.shape[0] == 0:
                break
            dists = np.linalg.norm(rest[:, None, :] - np.array(selected)[None, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            next_pt = rest[np.argmax(min_dists)]
            selected.append(next_pt)

        selected = np.array(selected)
        if selected.shape[0] < M:
            pad = pts[np.random.choice(N, M - selected.shape[0], replace=False)]
            selected = np.concatenate([selected, pad], axis=0)

        all_samples.append(selected[:M])

    return np.stack(all_samples, axis=0)  # [B, M, 3]

