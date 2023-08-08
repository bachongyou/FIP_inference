import torch
from math import sqrt
from torch.nn import functional as f
import numpy as np

def trace_method(matrix):
    device = matrix.device
    m = matrix.transpose(0,1) # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

    q = torch.tensor(q).to(device)
    q *= 0.5 / torch.sqrt(t)
    return q

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def q2mat(quat):
    quat = quat / torch.norm(quat, dim=1, keepdim=True)
    angle = 2*torch.acos(quat[:,0]).unsqueeze(1)
    axis = quat[:,1:]/torch.norm(quat[:,1:],dim=1,keepdim=True)
    aa = angle*axis
    return (batch_rodrigues(aa))

def batch_quat_mul(q1,q2):
#     B*J*4
    w = q1[:,:,0]*q2[:,:,0]-q1[:,:,1]*q2[:,:,1]-q1[:,:,2]*q2[:,:,2]-q1[:,:,3]*q2[:,:,3]
    x = q1[:,:,0]*q2[:,:,1]+q1[:,:,1]*q2[:,:,0]+q1[:,:,2]*q2[:,:,3]-q1[:,:,3]*q2[:,:,2]
    y = q1[:,:,0]*q2[:,:,2]-q1[:,:,1]*q2[:,:,3]+q1[:,:,2]*q2[:,:,0]+q1[:,:,3]*q2[:,:,1]
    z = q1[:,:,0]*q2[:,:,3]+q1[:,:,1]*q2[:,:,2]-q1[:,:,2]*q2[:,:,1]+q1[:,:,3]*q2[:,:,0]
    return torch.cat([w.unsqueeze(2),x.unsqueeze(2),y.unsqueeze(2),z.unsqueeze(2)],dim=2)

def quat_mul(q1,q2):
    w = q1[:,0]*q2[:,0]-q1[:,1]*q2[:,1]-q1[:,2]*q2[:,2]-q1[:,3]*q2[:,3]
    x = q1[:,0]*q2[:,1]+q1[:,1]*q2[:,0]+q1[:,2]*q2[:,3]-q1[:,3]*q2[:,2]
    y = q1[:,0]*q2[:,2]-q1[:,1]*q2[:,3]+q1[:,2]*q2[:,0]+q1[:,3]*q2[:,1]
    z = q1[:,0]*q2[:,3]+q1[:,1]*q2[:,2]-q1[:,2]*q2[:,1]+q1[:,3]*q2[:,0]
    return torch.stack([w,x,y,z],dim=1)

def batch_w2l_q(quat):
    # quat:[b,15,4]
    parents_idx = [-1,-1,-1,0,1,2,5,6,6,6,7,8,9,11,12]
    result = []
    for i,j in enumerate(parents_idx):
        if j == -1:
            result.append(quat[:,i].numpy())
        else:
            q_i = torch.cat([quat[:,j,0].unsqueeze(1),-quat[:,j,1:]],dim=1)
            result.append(quat_mul(q_i,quat[:,i]).numpy())
    return torch.tensor(result).transpose(0,1)

def batch_w2l_aa(quat):
    # quat:[b,15,4]
    parents_idx = [-1,-1,-1,0,1,2,5,6,6,6,7,8,9,11,12]
    result = []
    for i,j in enumerate(parents_idx):
        if j == -1:
            axis = quat[:,i,1:4]/torch.norm(quat[:,i,1:4],dim=1,keepdim=True)
            aa = axis*quat[:,i,0].unsqueeze(1)
        else:
            q_i = torch.cat([quat[:,j,0].unsqueeze(1),-quat[:,j,1:]],dim=1)
            q = quat_mul(q_i,quat[:,i])
            axis = q[:,1:4]/torch.norm(q[:,1:4],dim=1,keepdim=True)
            aa = axis*q[:,0].unsqueeze(1)
        result.append(aa)
    return torch.stack(result).transpose(0,1)

def batch_mat2q(mat):
    res = []
    for m in mat:
        res.append(trace_method(m))
    return torch.stack(res)

def q2aa(quat):
    angle = 2*torch.acos(quat[:,0]).unsqueeze(1)
    axis = quat[:,1:]/torch.norm(quat[:,1:],dim=1,keepdim=True)
    return (angle*axis)

def aa2q(aa):
    angle = torch.norm(aa,dim=1,keepdim=True)
    axis = aa/angle
    cos = torch.cos(angle/2)
    sin = torch.sin(angle/2)
    return torch.cat([cos,sin*axis],dim=1)

def weight_loss(pred,label, mode='raw'):
    """
    :param pred: [b,19,3]
    :param label: [b,19,3]
    :return: weighted loss
    """
    level = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16],[17,18]]
    weight = [18,15,12,9,6,3,1]
    for i in range(7):
        pred[:,level[i]] = sqrt(weight[i])*pred[:,level[i]]
        label[:,level[i]] = sqrt(weight[i])*label[:,level[i]]
    if mode == "raw":
        return pred, label
    return f.mse_loss(pred,label)

def forward_kinematics(body, pose):
    """
    :param body: [19,3]
    :param pose: [b,15,3,3]
    :return: joints[b,19,3]
    """
    b = pose.size(0)
    body = body.unsqueeze(0).repeat(b, 1, 1)
    p_tree = [[-1, -1, -1], [0, 1, 2], [3, 4, 5], [6, 6, 6], [7, 8, 9], [11, 12], [13, 14]]
    k_tree = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16], [17, 18]]
    x1 = body[:, k_tree[0]]
    x2 = x1 + torch.bmm(pose[:, p_tree[1]].reshape(b * 3, 3, 3),
                        body[:, k_tree[1]].reshape(b * 3, 3, 1)).reshape(b, 3, 3)
    x3 = x2 + torch.bmm(pose[:, p_tree[2]].reshape(b * 3, 3, 3),
                        body[:, k_tree[2]].reshape(b * 3, 3, 1)).reshape(b, 3, 3)
    x4 = x3[:, [2, 2, 2]] + torch.bmm(pose[:, p_tree[3]].reshape(b * 3, 3, 3),
                                      body[:, k_tree[3]].reshape(b * 3, 3, 1)).reshape(b, 3, 3)
    x5 = x4 + torch.bmm(pose[:, p_tree[4]].reshape(b * 3, 3, 3),
                        body[:, k_tree[4]].reshape(b * 3, 3, 1)).reshape(b, 3, 3)
    x6 = x5[:, [1, 2]] + torch.bmm(pose[:, p_tree[5]].reshape(b * 2, 3, 3),
                                   body[:, k_tree[5]].reshape(b * 2, 3, 1)).reshape(b, 2, 3)
    x7 = x6 + torch.bmm(pose[:, p_tree[6]].reshape(b * 2, 3, 3),
                        body[:, k_tree[6]].reshape(b * 2, 3, 1)).reshape(b, 2, 3)
    return torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=1)

def glb2local(glb_pose):
    global_full_pose = torch.eye(3, device=glb_pose.device).repeat(glb_pose.shape[0], 24, 1, 1)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
               16, 17, 18, 19]
    global_full_pose[:, [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]] = glb_pose
    for i in range(len(parents)-1 , 3 , -1):
        p = global_full_pose[:,parents[i]]
        global_full_pose[:,i] = p.transpose(-1,-2) @ global_full_pose[:,i]
    global_full_pose[:, [0, 7, 8, 10, 11, 20, 21,22,23]] = torch.eye(3, device=glb_pose.device)
    return global_full_pose


def batch2euler(rot_vecs, deg=True, unity=True, epsilon=1e-8, dtype=torch.float32):
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rotmat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    x = torch.atan2(-rotmat[:, 1, 2],
                    torch.sqrt(rotmat[:, 1, 0] * rotmat[:, 1, 0] + rotmat[:, 1, 1] * rotmat[:, 1, 1])).unsqueeze(1);
    y = torch.atan2(rotmat[:, 0, 2], rotmat[:, 2, 2]).unsqueeze(1);
    z = torch.atan2(rotmat[:, 1, 0], rotmat[:, 1, 1]).unsqueeze(1);
    if unity:
        y, z = -y, -z
    if deg:
        x, y, z = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return torch.cat([x, y, z], dim=1)


def mat2euler(rotmat, deg=True, unity=True, epsilon=1e-8, dtype=torch.float32):
    x = torch.atan2(-rotmat[:, 1, 2],
                    torch.sqrt(rotmat[:, 1, 0] * rotmat[:, 1, 0] + rotmat[:, 1, 1] * rotmat[:, 1, 1])).unsqueeze(1);
    y = torch.atan2(rotmat[:, 0, 2], rotmat[:, 2, 2]).unsqueeze(1);
    z = torch.atan2(rotmat[:, 1, 0], rotmat[:, 1, 1]).unsqueeze(1);
    if unity:
        y, z = -y, -z
    if deg:
        x, y, z = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return torch.cat([x, y, z], dim=1)


def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)


def vector_cross_matrix(x: torch.Tensor):
    r"""
    Get the skew-symmetric matrix :math:`[v]_\times\in so(3)` for each vector3 `v`. (torch, batch)

    :param x: Tensor that can reshape to [batch_size, 3].
    :return: The skew-symmetric matrix in shape [batch_size, 3, 3].
    """
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)


def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r
