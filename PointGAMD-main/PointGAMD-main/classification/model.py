import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from gp_pcs import gppcs_sample_indices

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src: [B, N, C], dst: [B, M, C] -> dist: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx, device_index=None):
    """
    Gathers points using the provided indices.
    Optimized version with better memory management.
    """
    assert points.dim() == 3, f"points should be [B, C, N] or [B, N, C], got {points.shape}"
    B = points.shape[0]

    # If input is [B, C, N], transpose to [B, N, C]
    if points.shape[1] <= 3:
        points = points.transpose(1, 2).contiguous()

    N = points.shape[1]
    C = points.shape[2]

    # Clamp indices to valid range
    idx = torch.clamp(idx, 0, N-1)

    # Efficient batched indexing
    batch_indices = torch.arange(B, device=points.device).view(-1, 1).expand(-1, idx.shape[1])
    new_points = points[batch_indices, idx]  # [B, S, C]
    return new_points

def farthest_point_sample(xyz, npoint, index):
    """
    Optimized FPS implementation with better memory management.
    """
    B, N, C = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        # Vectorized centroid computation
        centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)  # [B, 1, 3]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # [B, N]
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
    return centroids

def gppcs_sample(xyz, npoint, k_neighbors=16):
    """
    GP-PCS sampling function that returns indices.
    """
    # xyz: [B, N, 3]
    indices = gppcs_sample_indices(xyz, npoint, k_neighbors)
    return indices

class LocalGrouper(nn.Module):
    def __init__(self, groups, kneighbors, use_xyz=True, normalize='center', 
                 index=0, sampling_method='fps'):
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.k = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize
        self.index = index
        self.sampling_method = sampling_method  # 'fps' or 'gppcs'

    def forward(self, xyz, features):
        """
        Args:
            xyz: [B, 3, N]
            features: [B, D, N]
        Returns:
            new_xyz: [B, 3, groups]
            new_features: [B, D+3, groups, k]
        """
        B, _, N = xyz.shape
        S = self.groups
        D = features.shape[1]

        # === Step 1: Sampling (FPS or GP-PCS) ===
        xyz_t = xyz.transpose(1, 2).contiguous()  # [B, N, 3]
        
        if self.sampling_method == 'gppcs':
            sample_idx = gppcs_sample(xyz_t, S)  # [B, S]
        else:  # fps
            sample_idx = farthest_point_sample(xyz_t, S, self.index)  # [B, S]

        # Ensure indices are valid
        sample_idx = torch.clamp(sample_idx, 0, N-1)

        # === Step 2: Gather sampled points ===
        new_xyz = index_points(xyz, sample_idx, self.index)  # [B, S, 3]
        new_xyz = new_xyz.transpose(1, 2).contiguous()  # [B, 3, S]

        # === Step 3: Efficient KNN Grouping ===
        with torch.no_grad():
            # Use cdist for efficient distance computation
            new_xyz_t = new_xyz.transpose(1, 2)  # [B, S, 3]
            dists = torch.cdist(new_xyz_t, xyz_t)  # [B, S, N]
            
            # Get k nearest neighbors
            knn_idx = dists.topk(self.k, dim=-1, largest=False)[1]  # [B, S, k]

        # === Step 4: Vectorized feature gathering ===
        # Flatten indices for efficient gathering
        batch_size, num_groups, k = knn_idx.shape
        
        # Create batch indices
        batch_indices = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, S, self.k)
        
        # Gather XYZ coordinates
        xyz_gathered = xyz_t[batch_indices, knn_idx]  # [B, S, k, 3]
        grouped_xyz = xyz_gathered.permute(0, 3, 1, 2)  # [B, 3, S, k]
        
        # Gather features
        features_t = features.transpose(1, 2).contiguous()  # [B, N, D]
        features_gathered = features_t[batch_indices, knn_idx]  # [B, S, k, D]
        grouped_features = features_gathered.permute(0, 3, 1, 2)  # [B, D, S, k]

        # === Step 5: Concatenate features ===
        if self.use_xyz:
            new_features = torch.cat([grouped_features, grouped_xyz], dim=1)  # [B, D+3, S, k]
        else:
            new_features = grouped_features

        # === Step 6: Normalization ===
        if self.normalize == 'center':
            new_features[:, -3:] -= new_xyz.view(B, 3, S, 1)  # Center xyz part
        elif self.normalize == 'anchor':
            anchor = new_features[:, -3:, :, 0].unsqueeze(-1)
            new_features[:, -3:] -= anchor

        return new_xyz, new_features

class OptimizedResBlock(nn.Module):
    """Optimized residual block with GroupNorm for better training stability."""
    def __init__(self, channels, groups=8):
        super(OptimizedResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, 1, bias=False)
        # Use GroupNorm for better stability with small batch sizes
        self.norm1 = nn.GroupNorm(min(groups, channels), channels)
        self.norm2 = nn.GroupNorm(min(groups, channels), channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return self.activation(out)

class OptimizedResBlock2D(nn.Module):
    """Optimized 2D residual block."""
    def __init__(self, channels, groups=8):
        super(OptimizedResBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm1 = nn.GroupNorm(min(groups, channels), channels)
        self.norm2 = nn.GroupNorm(min(groups, channels), channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return self.activation(out)

class OptimizedOperation(nn.Module):
    """Optimized Operation module with reduced complexity and better efficiency."""
    def __init__(self, inp_feat, out_feat, k_neighbours, new_points, index, sampling_method='fps'):
        super(OptimizedOperation, self).__init__()
        self.out_feat = out_feat
        self.sampling_method = sampling_method

        # Use optimized grouper
        self.group = LocalGrouper(groups=new_points, kneighbors=k_neighbours, 
                                  use_xyz=True, normalize='center', index=index,
                                  sampling_method=sampling_method)

        # Simplified architecture with fewer parameters
        hidden_dim = out_feat // 2
        
        # === Shared feature extraction ===
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(inp_feat + 3, hidden_dim, 1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            OptimizedResBlock2D(hidden_dim),
        )
        
        # === Attention mechanism (simplified) ===
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # === Output projection ===
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, out_feat, 1, bias=False),
            nn.GroupNorm(min(8, out_feat), out_feat),
            nn.ReLU(inplace=True),
            OptimizedResBlock(out_feat)
        )

    def forward(self, xyz, feat):
        """
        Args:
            xyz: [B, 3, N]
            feat: [B, D, N]
        Returns:
            new_xyz: [B, 3, S]
            x: [B, out_feat, S]
        """
        new_xyz, grouped_feat = self.group(xyz, feat)  # [B, D+3, S, k]
        
        # Feature extraction
        x = self.feature_extractor(grouped_feat)  # [B, hidden_dim, S, k]
        
        # Attention-based pooling
        attention_weights = self.attention(x)  # [B, 1, S, k]
        x = x * attention_weights  # Weighted features
        x = torch.sum(x, dim=-1)  # Pool over neighbors: [B, hidden_dim, S]
        
        # Output projection
        x = self.output_proj(x)  # [B, out_feat, S]
        
        return new_xyz, x

class OptimizedClassificationHead(nn.Module):
    """Optimized classification head with better regularization."""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(OptimizedClassificationHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.layers(x)

class Model(nn.Module):
    def __init__(self, classes, k_neighbours, index, sampling_method='fps'):
        super(Model, self).__init__()
        self.sampling_method = sampling_method
        
        # === Initial feature extraction ===
        self.initial_features = nn.Sequential(
            nn.Conv1d(5, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        
        # === Hierarchical feature extraction ===
        self.operation1 = OptimizedOperation(64, 256, k_neighbours, 512, index, sampling_method)
        self.operation2 = OptimizedOperation(256, 512, k_neighbours, 256, index, sampling_method)
        
        # === Final feature extraction ===
        self.final_features = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True)
        )
        
        # === Classification head ===
        self.classification = OptimizedClassificationHead(1024, classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, point_inputs):
        """
        Args:
            point_inputs: [B, 3, N]
        Returns:
            outputs: [B, num_classes]
            embb: [B, 1024]
        """
        # === Feature engineering ===
        point_inputs_t = point_inputs.transpose(1, 2)  # [B, N, 3]
        
        # In-plane and out-of-plane distances
        in_plane_distances = torch.norm(point_inputs_t[:, :, :2], dim=2, keepdim=True)
        out_plane_distances = torch.abs(point_inputs_t[:, :, 2:3])
        
        # Combine features: [B, 5, N]
        features = torch.cat([
            point_inputs,
            in_plane_distances.transpose(1, 2),
            out_plane_distances.transpose(1, 2)
        ], dim=1)
        
        # === Initial feature extraction ===
        features = self.initial_features(features)  # [B, 64, N]
        
        # === Hierarchical processing ===
        point_inputs, features = self.operation1(point_inputs, features)  # [B, 3, 512], [B, 256, 512]
        point_inputs, features = self.operation2(point_inputs, features)  # [B, 3, 256], [B, 512, 256]
        
        # === Final feature extraction ===
        features = self.final_features(features)  # [B, 1024, 256]
        
        # === Global pooling ===
        embb = torch.max(features, dim=-1)[0]  # [B, 1024]
        
        # === Classification ===
        outputs = self.classification(embb)  # [B, num_classes]
        
        return outputs, embb

# Keep the old classes for backward compatibility
resblock = OptimizedResBlock
k_residual_block = OptimizedResBlock2D
Operation = OptimizedOperation
classification_network = OptimizedClassificationHead

if __name__ == '__main__':
    # Test both sampling methods
    print("Testing FPS model...")
    model_fps = Model(classes=10, k_neighbours=10, index=0, sampling_method='fps')
    
    print("Testing GP-PCS model...")
    model_gppcs = Model(classes=10, k_neighbours=10, index=0, sampling_method='gppcs')
    
    # Test input
    test_input = torch.randn(2, 3, 1024)
    
    print("FPS model forward pass...")
    out1, emb1 = model_fps(test_input)
    print(f"FPS - Output shape: {out1.shape}, Embedding shape: {emb1.shape}")
    
    print("GP-PCS model forward pass...")
    out2, emb2 = model_gppcs(test_input)
    print(f"GP-PCS - Output shape: {out2.shape}, Embedding shape: {emb2.shape}")


