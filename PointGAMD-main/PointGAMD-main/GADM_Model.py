import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
from torch.nn.parameter import Parameter

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx, index):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).cuda(index).view(  # .cuda(index)
        view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, index):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).cuda(
        index)  # .cuda(index)
    distance = torch.ones(B, N).cuda(index) * 1e10  # .cuda(index)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).cuda(
        index)  # .cuda(index)
    batch_indices = torch.arange(
        B, dtype=torch.long)  # .cuda(index)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    local_distance, group_idx = torch.topk(
        sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx, local_distance

class LocalGrouper(nn.Module):
    def __init__(self, feat_dim, groups, lamda, k_n, index, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.lamda = lamda
        self.index = index
        self.k_n = k_n
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, feat_dim]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, feat_dim]))
        self.affine_alpha_p = nn.Parameter(torch.ones([1, 1, 1, feat_dim]))
        self.affine_beta_p = nn.Parameter(torch.zeros([1, 1, 1, feat_dim]))

    def forward(self, xyz, feats):
        xyz = xyz.transpose(1, 2).contiguous()  # xyz [batch, points , xyz]
        feats = feats.transpose(1, 2).contiguous()  # xyz [batch, points , d]

        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()

        fps_idx = farthest_point_sample(xyz, self.groups, self.index).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]

        # pi, pj
        new_xyz = index_points(xyz, fps_idx, self.index)  # [B, npoint, 3]
        new_feats = index_points(feats, fps_idx, self.index)  # [B, npoint, d]

        # In point space
        idx,_ = knn_point(self.k_n, xyz, new_xyz)
        grouped_feats = index_points(
            feats, idx, self.index)  # [B, npoint, k, d]
        # Normalization
        mean = new_feats.unsqueeze(dim=-2)
        std = torch.std((grouped_feats-mean).reshape(B, -1),
                        dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        grouped_feats = (grouped_feats-mean)/(std + 1e-5)
        grouped_feats = self.affine_alpha_p*grouped_feats + self.affine_beta_p
        point_feats = torch.cat([grouped_feats, grouped_feats-new_feats.view(
            B, S, 1, -1).repeat(1, 1, self.k_n, 1)], dim=-1)

        new_feats = new_feats.permute(0, 2, 1).contiguous()
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        point_feats = point_feats.permute(0, 3, 1, 2).contiguous()

        return new_xyz, new_feats, point_feats 
class resblock(nn.Module):
    def __init__(self,
                 out_dims):
        super(resblock, self).__init__()
        self.Conv1 = nn.Conv1d(out_dims, out_dims, 1, 1)
        self.Conv2 = nn.Conv1d(out_dims, out_dims, 1, 1)

        self.batch1 = nn.BatchNorm1d(out_dims)
        self.batch2 = nn.BatchNorm1d(out_dims)

    def forward(self, inputs):
        x = self.batch1(self.Conv1(inputs))
        x = self.batch2(self.Conv2(x))
        # Res connection
        x = F.relu(inputs + x)
        # x -> G(fi)
        return x


class k_residual_block(nn.Module):
    def __init__(self,
                 out_dim):
        super(k_residual_block, self).__init__()
        self.Conv1 = nn.Conv2d(out_dim, out_dim, 1, 1)
        self.Conv2 = nn.Conv2d(out_dim, out_dim, 1, 1)

        self.batch1 = nn.BatchNorm2d(out_dim)
        self.batch2 = nn.BatchNorm2d(out_dim)

    def forward(self, inputs):
        x = self.batch1(self.Conv1(inputs))
        x = self.batch2(self.Conv2(x))
        # Res connection
        x = F.relu(inputs + x)
        # x -> G(fi)
        return x


class Operation(nn.Module):
    def __init__(self,
                 inp_feat,
                 out_feat,
                 k_neighbours,
                 new_points,
                 index
                 ):
        super(Operation, self).__init__()
        self.out_feat = out_feat

        self.group = LocalGrouper(
            inp_feat, new_points, 0.25, k_neighbours, index)

        self.pre_conv1 = nn.Conv1d(inp_feat, out_feat//2, 1, 1)
        self.pre_batchn1 = nn.BatchNorm1d(out_feat//2)
        self.k_conv1 = nn.Conv1d(out_feat//2, out_feat//2, 1, 1)
        self.k_batchn1 = nn.BatchNorm1d(out_feat//2)
        self.q_conv1 = nn.Conv1d(out_feat//2, out_feat//2, 1, 1)
        self.q_batchn1 = nn.BatchNorm1d(out_feat//2)
        self.v_conv1 = nn.Conv1d(out_feat//2, out_feat//2, 1, 1)
        self.v_batchn1 = nn.BatchNorm1d(out_feat//2)
        self.post_conv1 = nn.Conv1d(out_feat//2, out_feat, 1, 1)
        self.post_batchn1 = nn.BatchNorm1d(out_feat)

        self.residual1 = resblock(out_feat//2)
        self.residual2 = k_residual_block(out_feat//2)

        self.pre_conv2 = nn.Conv2d(inp_feat*2, out_feat//2, 1, 1)
        self.pre_batchn2 = nn.BatchNorm2d(out_feat//2)
        self.post_conv2 = nn.Conv2d(out_feat//2, out_feat, 1, 1)
        self.post_batchn2 = nn.BatchNorm2d(out_feat)

        self.multihead = nn.MultiheadAttention(embed_dim =out_feat//2, num_heads =1,batch_first = True)

        self.fin_conv = nn.Conv1d(out_feat*2, out_feat, 1, 1)
        self.fin_batch = nn.BatchNorm1d(out_feat)

    def forward(self, xyz, feat):
        # xyz -> B, 3, N
        # feat -> B, D, N

        new_xyz, inputs, point_feats = self.group(xyz, feat)

        res_x = F.relu(self.pre_batchn1(self.pre_conv1(inputs)))
        value = F.relu(self.v_batchn1(self.v_conv1(res_x)))
        key = F.relu(self.k_batchn1(self.k_conv1(res_x)))
        query = F.relu(self.q_batchn1(self.q_conv1(res_x)))
        
        x,_ = self.multihead(query.transpose(1,2), key.transpose(1,2) ,value.transpose(1,2))
        x = x.transpose(1,2)
        
        x = F.relu(x + res_x)
        x = self.residual1(x)
        x = F.relu(self.post_batchn1(self.post_conv1(x)))
        
        x1 = F.relu(self.pre_batchn2(self.pre_conv2(point_feats)))
        x1 = self.residual2(x1)
        x1 = F.relu(self.post_batchn2(self.post_conv2(x1)))
        x1 = torch.max(x1, dim =-1)[0]

        x = torch.concat([x,x1], dim=1)
        x = F.relu(self.fin_batch(self.fin_conv(x)))

        return new_xyz, x
# [rest of the code remains the same]
class classification_network(nn.Module):  # pooled features -> num_classes
    def __init__(self, i_feature, num_classes):
        super(classification_network, self).__init__()
        
        self.mlp_layer1 = nn.Linear(i_feature, 512)
        self.mlp_layer2 = nn.Linear(512, 256)
        self.mlp_layer3 = nn.Linear(256, 128)
        self.mlp_layer4 = nn.Linear(128, 64)
        self.mlp_layer5 = nn.Linear(64, num_classes)

        # self.batchn0 = nn.BatchNorm1d(512)
        self.batchn1 = nn.BatchNorm1d(512)
        self.batchn2 = nn.BatchNorm1d(256)
        self.batchn3 = nn.BatchNorm1d(128)
        self.batchn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):

        x = F.relu(self.batchn1(self.mlp_layer1(inputs)))
        x = self.dropout(x)
        x = F.relu(self.batchn2(self.mlp_layer2(x)))
        x = F.relu(self.batchn3(self.mlp_layer3(x)))
        x = self.dropout(x)
        x = F.relu(self.batchn4(self.mlp_layer4(x)))

        x = self.mlp_layer5(x)

        return x

class Model_class(nn.Module):
    def __init__(self,
                 num_classes=40,
                 k_neighbours=16,  # Added default value
                 index=0):  # Added default value
        super(Model_class, self).__init__()  # Corrected class name

        self.initial_features = nn.Conv1d(5, 64, 1, 1)
        self.batchn = nn.BatchNorm1d(64)
        self.index = index
        self.k_n = k_neighbours

        self.operation1 = Operation(64, 256, k_neighbours, 512, index)
        self.operation2 = Operation(256, 512, k_neighbours, 256, index)

        self.final_features = nn.Conv1d(512, 1024, 1, 1)
        self.final_batchn = nn.BatchNorm1d(1024)

        self.classification = classification_network(1024, num_classes)

    def forward(self, point_inputs):
        point_inputs = point_inputs.transpose(1,2)
        
        in_plane_distances = torch.norm(point_inputs[:, :, :2], dim=2, keepdim=True)
        out_plane_distances = torch.abs(point_inputs[:, :, 2]).unsqueeze(2)
        point_inputs = point_inputs.transpose(1,2)
        features = torch.cat([point_inputs,
                              in_plane_distances.transpose(1,2),
                              out_plane_distances.transpose(1,2)], dim=1)

        features = F.relu(self.batchn(
            self.initial_features(features)))  # trans_matrix

        #Classification
        point_inputs, features = self.operation1(point_inputs, features)
        point_inputs, features = self.operation2(point_inputs, features)
        
        features = F.relu(self.final_batchn(
            self.final_features(features)))
        embb = torch.max(features, dim=-1)[0]

        #features = self.classification(embb)

        return self.classification(embb)
