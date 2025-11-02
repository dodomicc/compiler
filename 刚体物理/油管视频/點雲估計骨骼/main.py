
import trimesh
import numpy as np
import networkx as nx

import jax.numpy as jnp
import time
from jax import *
from bone_estimation_loss import *
from utils import *

from scipy.spatial import cKDTree
from trimesh.proximity import ProximityQuery
scene = trimesh.load('./scene.gltf')
if isinstance(scene, trimesh.Scene):
    meshes = [geometry for geometry in scene.geometry.values()]
else:
    meshes = [scene]

all_faces = np.vstack([mesh.faces for mesh in meshes])
combined_mesh = trimesh.util.concatenate(meshes)
pq = ProximityQuery(combined_mesh)
sdf_values = pq.signed_distance(np.random.rand(1000,3))
print(sdf_values)
sdf_values = pq.signed_distance(np.random.rand(1000,3))
print(sdf_values)
sdf_values = pq.signed_distance(np.random.rand(1000,3))
print(sdf_values)

# def get_query_tree(all_vertices: np.ndarray)->cKDTree:
#     return cKDTree(all_vertices)

# def get_vertice_nors(all_vertices:np.ndarray,all_faces: np.ndarray)->np.ndarray:
#     vertices2faces = np.zeros((all_vertices.shape[0],3),dtype = np.uint32 )
#     vertices2nors = np.zeros((all_vertices.shape[0],3),dtype = np.float64 )
#     for face in all_faces:
#         vertices2faces[face[0],:] = face
#         vertices2faces[face[1],:] = face
#         vertices2faces[face[2],:] = face
#     for vertice in range(all_vertices.shape[0]):
#         point1 = all_vertices[vertices2faces[vertice,0],:]
#         point2 = all_vertices[vertices2faces[vertice,1],:]
#         point3 = all_vertices[vertices2faces[vertice,2],:]
#         nor = np.cross(point2 - point1, point3 - point1)
#         nor = nor/(np.linalg.norm(nor)+1e-6)
#         vertices2nors[vertice] = nor
#     return vertices2nors

# def get_signed_distance(point: np.ndarray, query_tree: cKDTree, vertices:np.ndarray, nors:np.ndarray)->float:
#     dist, index = query_tree.query(point)
#     dir = point - vertices[index]
#     return np.sign(np.dot(nors[index], dir)) * dist

# def get_distance(point: np.ndarray, query_tree: cKDTree)->float:
#     dist, _ = query_tree.query(point)
#     return dist







# def get_all_inside_voxels(interval_len:float, interval_range: np.ndarray, query_tree: cKDTree,vertices: np.ndarray, nors: np.ndarray)->np.ndarray:
#     min_coord = np.floor(interval_range[0,:]/interval_len).astype(np.int32)
#     max_coord = np.ceil(interval_range[1,:]/interval_len).astype(np.int32)
#     inside_voxels = []
#     size  = max_coord - min_coord
#     size = size[0] * size[1] * size[2]
#     s = 0
#     last_progress = np.round(-100,2)
#     for i in range(min_coord[0],max_coord[0]):
#         for j in range(min_coord[1],max_coord[1]):
#             for k in range(min_coord[2],max_coord[2]):
#                 cur_progress = np.round(s/size* 100,2)
#                 if(cur_progress - last_progress >2):
#                     last_progress = cur_progress
#                     print(f"內部體素提取進度:{cur_progress}%")     
#                 voxel_cen = np.array([i,j,k]) + 0.5
#                 voxel_cen = voxel_cen * interval_len
#                 if(get_signed_distance(voxel_cen, query_tree,vertices,nors)<0.):
#                     inside_voxels.append(voxel_cen)
#                 s = s + 1
#     print(f"內部體素數量：{len(inside_voxels)}")
#     return np.vstack(inside_voxels)

# def get_all_bone_voxels(interval_len:float, inside_voxels:np.ndarray, query_tree: cKDTree, vertices: np.ndarray, nors: np.ndarray) -> np.ndarray:
#     bone_voxels = []
#     size = inside_voxels.shape[0]
#     s = 0
#     last_progress = np.round(-100.,2)
#     for inside_voxel in inside_voxels:
#         cur_progress = np.round(s/size*100,2)
#         if(cur_progress - last_progress >2):
#             last_progress = cur_progress
#             print(f"骨骼體素提取進度:{cur_progress}%") 
#         dist = 0
#         cen_dist = get_signed_distance(inside_voxel,query_tree,vertices,nors)
#         for i in range(-1,2):
#             for j in range(-1,2):
#                 for k in range(-1,2):
#                     cen = inside_voxel + interval_len * np.array([i,j,k]).astype(np.float64)
#                     dist = min(dist, get_signed_distance(cen,query_tree, vertices, nors))
#         if(cen_dist<=dist): 
#             bone_voxels.append(cen)
#         s = s+1
#     print(f"骨骼體素數量：{len(bone_voxels)}")
#     return np.array(bone_voxels)

# def connect_bone_voxels(bone_voxels: np.ndarray,query_tree: cKDTree, vertices: np.ndarray,nors: np.ndarray, interior_penalty: float,root_idx:int|None = None):
#     root = None
#     dist = 0
#     bone_voxel_set = set()
#     bone_voxel_remaining_set = set()
#     bones = np.zeros((bone_voxels.shape[0]-1,2), dtype=np.uint32)
    
#     if(root_idx is None):
#         root_idx = 0
#         s = 0
#         for bone_voxel in bone_voxels:
#             dist0 = get_distance(bone_voxel,query_tree)
#             bone_voxel_remaining_set.add(s)
#             if(dist0>dist):
#                 dist = dist0
#                 root = bone_voxel
#                 root_idx = s
#             s = s + 1
#     else:
#         root = bone_voxels[root_idx,:]

#     bone_voxel_set.add(root_idx)
#     bone_voxel_remaining_set.remove(root_idx)
#     last_progress = np.round(0.,2)
#     for i in range(bone_voxels.shape[0] -1):
#         change_idx = 0
#         dist = 10000
       
#         for entry1 in bone_voxel_set:
#             for entry2 in bone_voxel_remaining_set:
#                 dist0 = np.linalg.norm(bone_voxels[entry1] - bone_voxels[entry2])
#                 cen1 = (bone_voxels[entry1] *2./3.+ bone_voxels[entry2] * 1/3)
#                 cen2 = (bone_voxels[entry1] *1./3.+ bone_voxels[entry2] * 2/3)
            
#                 entry1entry2_cen_sdf = get_signed_distance(cen1,query_tree,vertices,nors) + get_signed_distance(cen2,query_tree,vertices,nors)
#                 dist0 += interior_penalty* entry1entry2_cen_sdf
#                 if(dist0 < dist):
#                     dist = dist0
#                     bones[i,:] = np.array([entry1,entry2])
#                     change_idx = entry2
#         bone_voxel_set.add(change_idx)
#         bone_voxel_remaining_set.remove(change_idx) 
#         cur_progress = np.round(i/(bone_voxels.shape[0] -1)* 100,2)
#         if(cur_progress - last_progress >2):
#             last_progress = cur_progress
#             print(f"連接分析完成率:{cur_progress}%")     
#     return root,bones
            
        
# def compute_sdf_grad(point: np.ndarray,delta: float, query_tree: cKDTree, vertices:np.ndarray, nors:np.ndarray)->np.ndarray:
#     e1 = np.array([1.,0.,0.])
#     e2 = np.array([0.,1.,0.])
#     e3 = np.array([0.,0.,1.])
#     sdf1 = get_signed_distance(point - delta * e1,query_tree,vertices,nors)
#     sdf2 = get_signed_distance(point + delta * e1,query_tree,vertices,nors)
#     sdf3 = get_signed_distance(point - delta * e2,query_tree,vertices,nors)
#     sdf4 = get_signed_distance(point + delta * e2,query_tree,vertices,nors)
#     sdf5 = get_signed_distance(point - delta * e3,query_tree,vertices,nors)
#     sdf6 = get_signed_distance(point + delta * e3,query_tree,vertices,nors)
#     grad = np.array([sdf2 - sdf1,sdf4-sdf3, sdf6 - sdf5])/(2. * delta)
#     return grad

# def bone_voxels_decrease_grad(points: np.ndarray,delta: float, query_tree: cKDTree, vertices:np.ndarray, nors:np.ndarray) ->np.ndarray:
#     res = np.zeros((points.shape[0],3))
#     for i in range(points.shape[0]):
#         res[i,:] = compute_sdf_grad(points[i,:],delta,query_tree,vertices,nors)
#     return res

# def trim_bones(bones:np.ndarray, bone_voxels: np.ndarray,min_threshold:float = 0.05,max_threshold:float = 2.)->np.ndarray:
#     G = nx.DiGraph()
#     for bone in bones:
#         G.add_edge(bone[0],bone[1])
#     leaf_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
#     print(len(leaf_nodes))
#     sub_paths = []
#     for leaf_node in leaf_nodes:
#         path = []
#         node = leaf_node
#         path.append(node)
#         while(True):
#             node = list(G.predecessors(node))[0]
#             path.insert(0,node)
#             if(len(list(G.successors(node)))>1 or G.in_degree(node) == 0) : break
#         sub_paths.append(path)
#     i = 0
#     for path in sub_paths:
#         len0 = 0
#         for j in range(len(path)):
#             if(j>0): len0 += np.linalg.norm(bone_voxels[path[j],:] - bone_voxels[path[j-1],:])
#         if(len0<min_threshold): G.remove_edge(path[0],path[1])
    
#         i = i+1
#     components = list(nx.weakly_connected_components(G))
#     largest_cc = max(components, key=len)
#     largest_sub_graph = G.subgraph(largest_cc)
#     leafs = [n for n in largest_sub_graph.nodes if largest_sub_graph.out_degree(n) == 0]
#     leafs_res = np.zeros((len(leafs),3))
#     for i in range(len(leafs_res)):
#         leafs_res[i,:] = bone_voxels[leafs[i],:]
#     edges = largest_sub_graph.edges()

#     bones = np.zeros((len(edges),2),dtype=np.uint32)
#     i=0
#     for u, v in edges:
#         bones[i,0] = u
#         bones[i,1] = v
#         i = i+1
#     return bones, leafs_res

# def minimize_sdf(bone_voxels: np.ndarray,query_tree: cKDTree, vertices: np.ndarray, nors: np.ndarray, total_steps: int, step:float)->np.ndarray:
#     last_progress = np.round(0.,2)
#     for i in range(total_steps):
#         bone_voxels = bone_voxels - step * bone_voxels_decrease_grad(bone_voxels,0.0001, query_tree,vertices,nors)
#         bone_voxels += step * 0.1 * (2 * np.random.rand(3) - 1)
#         cur_progress = np.round(i/total_steps* 100,2)
#         if(cur_progress - last_progress >2):
#             last_progress = cur_progress
#             print(f"骨骼體素sdf局部最小化優化進度:{cur_progress}%")
#     return bone_voxels

# # print(compute_sdf_grad(np.random.rand(3), 0.001, query_tree,all_vertices,nors))
# range0 = np.zeros((2,3))

# all_vertices = np.vstack([mesh.vertices for mesh in meshes])
# all_vertices -= np.mean(all_vertices,axis=0)
# # all_vertices *= 0.5

# # all_vertices *= 0.2
# range0[0,:] = np.min(all_vertices,axis=0) 
# range0[1,:] = np.max(all_vertices,axis=0) 
# max_axis_dist = np.max(range0[1,:] -range0[0,:] ) * 0.7
# cen = np.mean(all_vertices, axis = 0)
# print(cen, max_axis_dist)
# query_tree = get_query_tree(all_vertices)
# nors = get_vertice_nors(all_vertices,all_faces)
# step = 0.0005
# total_steps = 2000


# inside_voxels = get_all_inside_voxels(0.05,range0,query_tree,all_vertices,nors)
# bone_voxels = get_all_bone_voxels(0.05,inside_voxels,query_tree,all_vertices, nors)
# bone_voxels = minimize_sdf(bone_voxels,query_tree,all_vertices,nors,total_steps,step)
# root, bones = connect_bone_voxels(bone_voxels,query_tree,all_vertices,nors,4.)
# bones ,leafs= trim_bones(bones,bone_voxels, 0.5,1.)
# print(leafs)




    




# # print(bone_voxels.shape)


# import matplotlib.pyplot as plt


# # 建立圖與 3D 座標軸
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# for bone in bones:
#     i = bone[0]
#     j = bone[1]
#     x = [bone_voxels[i, 0], bone_voxels[j, 0]]
#     y = [bone_voxels[i, 1], bone_voxels[j, 1]]
#     z = [bone_voxels[i, 2], bone_voxels[j, 2]]

#     # ax.plot(x, y, z, c='g', linewidth=1)
#     # 起點
#     x0, y0, z0 = bone_voxels[i]
#     # 終點 - 起點
#     u = bone_voxels[j, 0] - x0
#     v = bone_voxels[j, 1] - y0
#     w = bone_voxels[j, 2] - z0
#     length0 = np.linalg.norm(np.array([u,v,w]))

#     # 畫箭頭 (紅色)
#     ax.quiver(x0, y0, z0, u, v, w, length=length0, normalize=True, color='r')


# xlim = (cen[0] - max_axis_dist, cen[0] + max_axis_dist)
# ylim = (cen[1] - max_axis_dist, cen[1] + max_axis_dist)
# zlim = (cen[2] - max_axis_dist, cen[2] + max_axis_dist)


# # 加上軸標籤（可選）
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# ax.set_zlim(zlim)
# ax.set_box_aspect([1, 1, 1])  # 顯示出來是等長


# plt.show()
