
# priorPatch_path = 'F:\WSI_Code\TransMIL-main\Camelyon\cluster'
# filterPatch_path = 'F:\WSI_Code\TransMIL-main\Camelyon\pt_files'


h5_path = 'F:\WSI_Code\\reslut_Camel_Patch\patches'
priorPatch_path = 'Camelyon\cluster'
filterPatch_path = 'Camelyon\pt_files'
# file_pattern = os.path.join(priorPatch_path, '*.pt')
# filter_pattern = os.path.join(filterPatch_path,'*.pt')
#
# def compute_similarity(W, centers):
#     # W shape: (N, 512), centers shape: (40, 512)
#     W_norm = F.normalize(W, dim=1)  # (N, 512)
#     C_norm = F.normalize(centers, dim=1)  # (40, 512)
#     sim = torch.matmul(W_norm, C_norm.T)  # (N, 40)
#     max_sim, _ = sim.max(dim=1)  # 每个patch对40中心的最大相似度
#     return max_sim  # shape: (N,)
#
#
# def retain_low_similarity_patches(patches, sim_scores, retain_ratio):
#     # 保留相似度最低的 retain_ratio 的 patch
#     N = len(sim_scores)
#     retain_num = int(N * retain_ratio)
#     sorted_indices = torch.argsort(sim_scores)  # 从低到高排序
#     retained_indices = sorted_indices[:retain_num]
#     return patches[retained_indices]
#
# cluster_centers = []
#
#
# for pt_file in sorted(glob.glob(file_pattern)):
#     feats = torch.load(pt_file)  # shape: (7258, 512)
#     mean_vec = feats.mean(dim=0)  # shape: (512,)
#     cluster_centers.append(mean_vec)
#
# # 拼成一个 (40, 512) 的张量
# cluster_centers_tensor = torch.stack(cluster_centers)
#
#
# for filter_file in sorted(glob.glob(filter_pattern)):
#      filter_feats = torch.load(filter_file)
#      mean_feat = filter_feats.mean(dim=0,keepdim = True)
#      sim_scores = compute_similarity(mean_feat,cluster_centers_tensor)
#      retain_rate = 0.2 + 0.5 * sim_scores
#      max_sim = compute_similarity(filter_feats, cluster_centers_tensor)
#      save_feature = retain_low_similarity_patches(filter_feats,max_sim,retain_rate)
#      save_name = os.path.basename(filter_file)
#      save_path = os.path.join('retain_feature_COAD',save_name)
#      torch.save(save_feature,save_path)


import os
import glob
import torch
import torch.nn.functional as F
import h5py

file_pattern = os.path.join(priorPatch_path, '*.pt')
filter_pattern = os.path.join(filterPatch_path, '*.pt')

def compute_similarity(W, centers):
    W_norm = F.normalize(W, dim=1)
    C_norm = F.normalize(centers, dim=1)
    sim = torch.matmul(W_norm, C_norm.T)
    max_sim, _ = sim.max(dim=1)
    return max_sim

def retain_low_similarity_patches(patches, sim_scores, retain_ratio):
    N = len(sim_scores)
    retain_num = int(N * retain_ratio)
    sorted_indices = torch.argsort(sim_scores)  # 从低到高
    retained_indices = sorted_indices[:retain_num]
    return retained_indices

# 得到 cluster centers
cluster_centers = []
for pt_file in sorted(glob.glob(file_pattern)):
    feats = torch.load(pt_file)  # (N, 512)
    mean_vec = feats.mean(dim=0) # (512,)
    cluster_centers.append(mean_vec)
cluster_centers_tensor = torch.stack(cluster_centers)

# 遍历 filterPatch 文件
for filter_file in sorted(glob.glob(filter_pattern)):
    filter_feats = torch.load(filter_file)   # (N, 512)
    mean_feat = filter_feats.mean(dim=0, keepdim=True)
    sim_scores = compute_similarity(mean_feat, cluster_centers_tensor)
    # print(sim_scores)
    retain_rate = 0.6 - 0.01 * sim_scores
    max_sim = compute_similarity(filter_feats, cluster_centers_tensor)

    # 得到保留索引
    retained_indices = retain_low_similarity_patches(filter_feats, max_sim, retain_rate)

    # 筛选特征
    save_feature = filter_feats[retained_indices]

    # 保存特征
    save_name = os.path.basename(filter_file)
    save_path = os.path.join('retain_feature_Ours_test', save_name)
    torch.save(save_feature, save_path)

    # --- 新增：保存坐标 ---
    # 找到对应 h5 文件
    h5_file = os.path.join(h5_path, save_name.replace('.pt', '.h5'))
    print(h5_file)
    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            coords = f['coords'][:]   # (N, 2)
            print(coords.shape)
        retained_coords = coords[retained_indices.cpu().numpy()]  # 筛选坐标

        # 保存到新 h5
        save_h5_path = os.path.join('retain_coords_Camel', save_name.replace('.pt', '.h5'))
        os.makedirs(os.path.dirname(save_h5_path), exist_ok=True)
        with h5py.File(save_h5_path, 'w') as f_out:
            f_out.create_dataset('coords', data=retained_coords)
    else:
        print(f"对应的坐标文件不存在: {h5_file}")

