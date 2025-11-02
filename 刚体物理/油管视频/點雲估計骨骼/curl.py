# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from curve_split import * 
# from scipy.spatial import cKDTree
# import time
# # 參數範圍




    
# def brute_force_segment_query(segments: np.ndarray, point: np.ndarray):
#     """
#     暴力方式：计算 point 到所有线段的投影点并返回最近的。
    
#     参数:
#     - segments: (N, 2, 2) array, 每条线段两个端点
#     - point: (2,) 要查询的点

#     返回:
#     - closest_proj: 最近点在路径上的投影
#     - min_dist: 最小距离
#     """
#     min_dist = 100
#     closest_proj = None

#     for i in range(segments.shape[0]-1):
#         a = segments[i]
#         b = segments[i+1]
#         ap = point - a
#         ab = b - a
#         t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
#         proj = a + t * ab
#         dist = np.linalg.norm(point - proj)
#         if dist < min_dist:
#             min_dist = dist
#             closest_proj = proj

#     return closest_proj, min_dist

# def generate_flower_curve(petal_count=5,  radius=10, num_points=20000, phase=0.0):
#     """
#     生成一条基于花瓣曲线的2D扰动曲线（开放的，不闭合）。

#     参数:
#     - petal_count: 花瓣数量（控制曲线频率）
#     - noise_scale: 噪声幅度
#     - radius: 基础半径
#     - num_points: 曲线采样点数
#     - phase: 起始相位（用于旋转花瓣）
    
#     返回:
#     - curve: (num_points, 2) 的 numpy 数组，表示 2D 曲线点
#     """
#     t = np.linspace(0,  np.pi, num_points, endpoint=False)  # 不闭合
#     r = radius * (1 + 0.3 * np.sin(petal_count * t + phase))

#     # 添加扰动
#     # r += noise_scale * np.random.randn(num_points)

#     x = r * np.cos(t)
#     y = r * np.sin(t)

#     curve = np.stack([x, y], axis=1)
#     return curve

# curve = generate_flower_curve()

# query_point = np.random.rand(2) * 100  -50

# curve2 = recursive_split_segments(curve,0.9999)
# print(curve2.shape)
# segmentKDTree = SegmentKDTree(curve2)
# kd_proj, kd_dist = segmentKDTree.query_point(query_point)


# time1 = time.time()
# for _ in range(1000):
#     segmentKDTree.query_point(query_point)
# print((time.time() - time1)/1000)

# time1 = time.time()
# for _ in range(1000):
#     brute_force_segment_query(segmentKDTree.segments, query_point)
# print((time.time() - time1)/1000)

# # 暴力法結果
# brute_proj, brute_dist = brute_force_segment_query(segmentKDTree.segments, query_point)

# # 對比誤差
# print("Query Point:", query_point)
# print("KDTree    -> proj:", kd_proj, "dist:", kd_dist)
# print("BruteForce-> proj:", brute_proj, "dist:", brute_dist)
# print("Diff in proj:", np.linalg.norm(kd_proj - brute_proj))
# print("Diff in dist:", abs(kd_dist - brute_dist))

import skeletor as sk
# mesh = sk.example_mesh()
# # >>> # To load and use your own mesh instead of the example mesh:
# # >>> # import trimesh as tm
# # >>> # mesh = tm.Trimesh(vertices, faces)  # or...
# # >>> # mesh = tm.load_mesh('mesh.obj')
# fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
# skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)

# <Skeleton(vertices=(1258, 3), edges=(1194, 2), method=wavefront)>














# # paths = trimesh.load_path()

# # final_segments = recursive_split_segments(final_segments,0.99)






# plt.figure(figsize=(8, 4))

# # # 原始曲線：灰色虛線
# # plt.plot(curve[:, 0], curve[:, 1], color='gray', linestyle = '-',linewidth = 0.2, label='Original Curve')

# for i in range(2, curve.shape[0]+1):
#     plt.plot(curve2[i-2:i, 0], curve2[i-2:i, 1], linewidth=1)
# plt.axis('equal')
# plt.title("Segmented Curve (1-level split)")
# plt.grid(True)
# plt.show()

# import jax
# import jax.numpy as jnp
# from flax import linen as nn
# class ConvInOut(nn.Module):
#     out_channels: int

#     @nn.compact
#     def __call__(self, x):
        
#         print("输入 shape:", x.shape)
#         conv = nn.Conv(features=self.out_channels, kernel_size=(3,3), padding='SAME')
#         x = conv(x)
#         print("卷积核 shape:", conv.variables['params']['kernel'].shape)
#         print("输出 shape:", x.shape)
#         return x

# # 假设输入是batch=1, 5x5大小，3个通道
# x = jnp.ones((1, 5, 5, 3))

# model = ConvInOut(out_channels=8)  # 输出通道数8
# variables = model.init(jax.random.PRNGKey(0), x)
# output = model.apply(variables, x)
