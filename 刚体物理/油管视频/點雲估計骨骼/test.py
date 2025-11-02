import numpy as np
from scipy import ndimage

import trimesh
from typing import * 
import matplotlib.pyplot as plt

from utils import * 
from scipy.spatial import cKDTree
import networkx as nx
from skimage.morphology import skeletonize
from skan import Skeleton
import  scipy.ndimage as ndi
from scipy import sparse
from numba import njit
from scipy.sparse.csgraph import *
from collections import deque
from numpy.linalg import *
import time
from copy import *
import re
# def find_pca_ratio(A:np.ndarray)->float:
#     A = A - np.mean(A,axis = 0)
#     symmetry_mat = A.T @ A
#     eign_vals, eigen_vecs = np.linalg.eigh(symmetry_mat)
#     return np.max(eign_vals)/np.sum(eign_vals), eigen_vecs @ (eign_vals.reshape((-1,1))/np.sum(eign_vals))

# def find_start_end_point(A: np.ndarray)->np.ndarray:
#     A0 = A.copy()
#     A = A - np.mean(A,axis = 0)
#     symmetry_mat = A.T @ Apip
#     eigen_vals, eigen_vecs = np.linalg.eigh(symmetry_mat)
#     main_dir = eigen_vecs[np.argmax(eigen_vals)]
#     proj = A @  main_dir
#     start = A0[np.argmin(proj),:]
#     end = A0[np.argmax(proj),:]
#     return start, end


    
import networkx as nx

def prunted_edge2end_points(prunted_edges: List)->np.ndarray:
    res = np.zeros((len(prunted_edges),2), dtype=np.int32)
    for i,prunted_edge in enumerate(prunted_edges):
        match = re.match(r"(\d+)-(\d+)", prunted_edge)
        if match:
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            res[i,0] = num1
            res[i,1] = num2
    return res

def get_center_node(end_points:np.ndarray)->int:
    graph  = nx.Graph()
    for edge in end_points:
        graph.add_edge(edge[0],edge[1])
        graph.add_edge(edge[1],edge[0])
    distance_square = np.zeros((1,len(graph.nodes)),dtype=np.uint32)    
    for i, node in enumerate(graph.nodes):
        visited = set()
        visiited_times = {}
        queue = [node]
        visiited_times[node] = 1
        leaf_times = {}
        while queue:
            node_from = queue.pop(len(queue)-1)
            flag = False
            for neighbor in graph.neighbors(node_from):
                if neighbor in visited:
                    continue
                else:
                    flag = True
                    queue.append(neighbor)
                    visited.add(neighbor)
                    visiited_times[neighbor] = visiited_times[node_from] + 1
            if(not flag):
                leaf_times[node_from] = visiited_times[node_from]
        distance_square[0,i] = np.sum(np.array(list(leaf_times.values())) ** 2)
    center_node = list(graph.nodes)[np.argmin(distance_square[0])]
    return graph, center_node

def cut_leaf(graph: nx.Graph, center_node: int, recursive_threshold:float =0.2, ratio_threshold:float = 0.2)->int:
    max_dist_to_leaf = {}
    directed_graph = nx.DiGraph()
    visited = set()
    visited.add(center_node)
    queue = [center_node]
    while queue:
        node_from = queue.pop(len(queue) - 1)
        for neighbor in graph.neighbors(node_from):
            if neighbor in visited:
                continue
            else:
                queue.append(neighbor)
                visited.add(neighbor)
                directed_graph.add_edge(node_from,neighbor)
    leaves = [node for node in directed_graph.nodes if directed_graph.degree[node] == 1]
    directed_graph_copy = deepcopy(directed_graph)
    visited = set()
    while(len(leaves)>0):
        for leaf in leaves:
            max_dist_to_leaf[leaf] = 0
            for neighbor in graph.neighbors(leaf):
                if neighbor in visited: 
                    max_dist_to_leaf[leaf] = max(1 + max_dist_to_leaf[neighbor], max_dist_to_leaf[leaf])
            visited.add(leaf)
            directed_graph.remove_node(leaf)
        leaves = [node for node in directed_graph.nodes if directed_graph.degree[node] == 1]
    if(len(list(directed_graph.nodes))>0):
        max_dist_to_leaf[center_node] = 0
        for neighbor in graph.neighbors(center_node):
            if neighbor in visited: 
                max_dist_to_leaf[center_node] = max(1 + max_dist_to_leaf[neighbor], max_dist_to_leaf[center_node])
        visited.add(center_node)
        directed_graph.remove_node(center_node)

    leaves = [node for node in directed_graph_copy.nodes if directed_graph_copy.degree[node] == 1]
    for leaf in leaves:
   
        junction_points = list(directed_graph_copy.predecessors(leaf))
        junction_point =  junction_points[0]
        intermediate_points = list(directed_graph_copy.successors(junction_point))
        dist_to_junction = 0
        for k in range(int(recursive_threshold * max_dist_to_leaf[center_node])):
          
            while(len(intermediate_points) == 1 and junction_point!=center_node):
                junction_points = list(directed_graph_copy.predecessors(junction_point))
                junction_point =  junction_points[0]
                intermediate_points = list(directed_graph_copy.successors(junction_point))
                dist_to_junction = dist_to_junction + 1
            ratio = dist_to_junction/max_dist_to_leaf[junction_point]
            if(ratio<ratio_threshold):  
                directed_graph_copy.remove_node(leaf)
                break
            junction_points = list(directed_graph_copy.predecessors(leaf))
            if(len(junction_points) == 0): 
                break
            else:
                junction_point =  junction_points[0]
                intermediate_points = list(directed_graph_copy.successors(junction_point))            
    prunted_edges = []
    queue = [center_node]
    while queue:
        node_from = queue.pop(len(queue) - 1)
        for neighbor in directed_graph_copy.successors(node_from):
            queue.append(neighbor)
            prunted_edges.append(f"{min(node_from,neighbor)}-{max(node_from,neighbor)}")
   
    return prunted_edges
        # print(dist_to_junction,max_dist_to_leaf[junction_point])
        # print(junction_point_successor)
        # print(list(directed_graph_copy.successors(junction_point)))


def bone_segments2Line(graph: nx.Graph, center_node: int, coords:dict, var_threshold:float = 0.25)->list:
    bones = {}
    leaves = {}
    vars = {}
    visited = []
    queue = []
    visited.append(center_node)
    queue.append(center_node)
    bones[center_node] = [center_node]
    while(len(queue)>0):
        node = queue.pop()
        count = 0
        for neighbor in graph.neighbors(node):
            if(not neighbor in visited): 
                visited.append(neighbor)
                queue.append(neighbor)
                bones[neighbor] =  bones[node] + [neighbor]
                count+=1
        if(count == 0): leaves[node] = bones[node]
    leaf2Vars = np.zeros((len(leaves),2),dtype=np.int32)
    for i, key in enumerate(leaves.keys()):
        path = np.array(list(leaves[key]))
        nodes = coords[path]
        vars[key] = np.var(np.linalg.norm(nodes - np.mean(nodes,axis = 0), axis =1))
        leaf2Vars[i,0] = key
        leaf2Vars[i,1] = vars[key]
    leaf2Vars = leaf2Vars[np.argsort(leaf2Vars[:,1])]
    vars0 = leaf2Vars[:,1]
    sigma = np.sqrt(np.var(vars0))
    threshold = np.mean(vars0) - 3 * sigma
    
    threshold = leaf2Vars[int(leaf2Vars.shape[0] * var_threshold),1]
    leaf2Vars = leaf2Vars[leaf2Vars[:,1]>threshold]
    
    for entry in leaf2Vars:
        print(len(leaves[entry[0]]))
    
    edges = []
    for leaf in leaf2Vars:
        path = leaves[leaf[0]]
        for i in range(1,len(path)):
            edges.append(f"{min(path[i-1], path[i])}-{ max(path[i-1], path[i])}")
    edges = list(set(edges))
 
    return edges

    

def find_bone_points(voxels:np.ndarray, pitch:float)->np.ndarray:
    min_voxels = np.min(voxels, axis=0)
    max_voxels = np.max(voxels, axis=0)
    size = (max_voxels - min_voxels)/pitch
    size = np.round(size).astype(np.uint32) + 1
    binry_image = np.zeros((size[0],size[1],size[2]), dtype=np.uint8)
    voxels_coord = np.round((voxels - min_voxels) / pitch).astype(np.uint32)  # (N,3)
    binry_image[voxels_coord[:,0], voxels_coord[:,1], voxels_coord[:,2]] = 1 
    skelton = skeletonize(binry_image)
    print("主骨骼提取完成")
    bones = Skeleton(skelton)
    print("骨骼分段完成")
    bone_coordinates = bones.coordinates
    bones_list = bones.paths_list()
    end_points = np.zeros((len(bones_list),2), dtype=np.int32)
    bone_segments = []
    bone_segments_dict = {}
    for i, bone in  enumerate( bones_list):
        bone_segment = bone_coordinates[bone]
        # bone_segments.append(min_voxels + bone_segment.astype(np.float64) * pitch)
        end_points[i,0] = bone[0]
        end_points[i,1] = bone[-1]
        bone_segments_dict[f"{min(bone[0],bone[-1])}-{max(bone[0],bone[-1])}"] = min_voxels + bone_segment.astype(np.float64) * pitch
    graph, center_node = get_center_node(end_points)
    prunted_edges = cut_leaf(graph,center_node,0.1, 0.1)
    prunted_endpoints = prunted_edge2end_points(prunted_edges)
    while(prunted_endpoints.shape[0] != end_points.shape[0]):
        end_points = prunted_endpoints
        graph, center_node = get_center_node(end_points)
        prunted_edges = cut_leaf(graph,center_node,0.1, 0.1)
        prunted_endpoints = prunted_edge2end_points(prunted_edges)
     
    prunted_edges = bone_segments2Line(graph,center_node,bone_coordinates, 0.1)
    print(len(prunted_edges))
    prunted_endpoints = prunted_edge2end_points(prunted_edges)

    for prunted_edge in prunted_edges:
        bone_segments.append(bone_segments_dict[prunted_edge])
    bone_points = min_voxels + np.argwhere(skelton).astype(np.float64) * pitch

    return bone_points, bone_segments

scene = trimesh.load('./scene.gltf')
combined_mesh = trimesh.util.concatenate(scene)
vertices = combined_mesh.vertices
cen = np.mean(vertices,axis =0)
gap = 0.6 * np.max(np.max(vertices,axis=0) - np.min(vertices,axis=0))
pitch = gap/512
voxelized = combined_mesh.voxelized(pitch=pitch) 
filled = voxelized.fill()
internal_voxels = filled.points  # shape: (N, 3)
query_point = internal_voxels
print("物體內部體素提取完成")


bone_points, bone_segments= find_bone_points(internal_voxels, pitch)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for bone_segment in bone_segments:
    ax.scatter(bone_segment[:,0],bone_segment[:,1],bone_segment[:,2], s= 1)
# 加上軸標籤（可選）
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([cen[0] -gap, cen[0] + gap])
ax.set_ylim([cen[1] -gap, cen[1] + gap])
ax.set_zlim([cen[2] -gap, cen[2] + gap])
ax.set_box_aspect([1, 1, 1])  # 顯示出來是等長
plt.show()



