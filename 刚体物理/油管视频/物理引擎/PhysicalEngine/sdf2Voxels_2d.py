import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import jax.numpy as jnp
import jax.nn as jnn
from jax import *
from typing import *
from constraint import *
import numpy as np
import networkx as nx
from ManimUtils.utils import *



def sort_voxels(arr: np.ndarray)->np.ndarray:
        arr_sorted_idx = np.lexsort((arr[:, 1], arr[:, 0]))
        sorted_arr = arr[arr_sorted_idx]
        return  sorted_arr

def is_equal(pair1: np.ndarray,pair2:np.ndarray)->int:
        if(np.all(np.floor(pair1) == np.floor(pair2))): 
            return 1
        else:
            return 0
    
def is_greater(pair1: np.ndarray,pair2:np.ndarray)-> int:
    sorted_arr = np.array(
       [ pair1,
        pair2]
    )
  
    return  np.lexsort((sorted_arr[:, 1], sorted_arr[:, 0]))[0]

def is_less(pair1: np.ndarray,pair2:np.ndarray)-> int:
    sorted_arr = np.array(
       [ pair1,
        pair2]
    )
    return  np.lexsort((sorted_arr[:, 1], sorted_arr[:, 0]))[1]


def find_pair_in_arr(arr:np.ndarray,pair:np.ndarray)->int:
    start = 0
    end = len(arr) -1
  
    if(is_equal(pair,arr[start])):
        return start
    if(is_equal(pair,arr[end])):
        return end
    if(is_less(pair,arr[start]) == 1):
        return -1
    if(is_greater(pair,arr[end])== 1):
        return -1
    while end - start>2:
        mid = int(np.floor((start+end)/2))
        if(is_equal(pair,arr[mid])):
            return mid
        if(is_greater(pair,arr[mid])):
            start = mid
        if(is_less(pair,arr[mid])):
            end = mid
    for i in range(start,end):
        if(is_equal(pair,arr[i])):  return i
    return -1

def get_all_edge_for_cube(cube: np.ndarray)->np.ndarray:
    initial_edge = np.concatenate([np.ceil(cube),np.ceil(cube)])
    edges = np.array(
        [
            [0.,0.,1.,0.],
            [0.,0.,0.,1.],
            [0.,1.,1.,1.],
            [1.,0.,1.,1.]
        ]
    )
    edges = edges + initial_edge
    return edges

def is_edges_connected(edge1: np.ndarray, edge2: np.ndarray, tol=1e-6) -> bool:
    # 拆解端点
    p1 = edge1[:2]
    p2 = edge1[2:]
    q1 = edge2[:2]
    q2 = edge2[2:]

    # 判断是否有任意端点相等（考虑浮点误差）
    return (
        np.allclose(p1, q1, atol=tol) or
        np.allclose(p1, q2, atol=tol) or
        np.allclose(p2, q1, atol=tol) or
        np.allclose(p2, q2, atol=tol)
    )

def get_all_adjacent_cube(cube: np.ndarray)->np.ndarray:
    
    nors = np.array(
        [
            [0.,-1.],
            [-1.,0.],
            [0.,1.],
            [1.,0.]
            
        ]
    )
    return cube + nors
       
def sdf2initial_voxels_2d(sdf:Callable,range0:np.ndarray,edge_length:float,threshold:float)->np.ndarray:
    range0 = jnp.array(range0)
    x_min, x_max = range0[0]
    y_min, y_max = range0[1]

    x_size = int(jnp.ceil((x_max - x_min) / edge_length))
    y_size = int(jnp.ceil((y_max - y_min) / edge_length))

    # 创建网格索引 [i,j]
    i_idx = jnp.arange(x_size)
    j_idx = jnp.arange(y_size)
    ii, jj = jnp.meshgrid(i_idx, j_idx, indexing="ij")  # shape = (x_size, y_size)

    # 计算中心点坐标
    cen_x = x_min + (ii + 0.5) * edge_length
    cen_y = y_min + (jj + 0.5) * edge_length
    centers = jnp.stack([cen_x, cen_y], axis=-1)  # shape = (x_size, y_size, 2)

    # 向量化评估 sdf
    sdf_vals = vmap(vmap(sdf))(centers)  # shape = (x_size, y_size)

    # 取绝对值小于阈值的点
    mask = jnp.abs(sdf_vals) < threshold

    # 提取满足条件的索引
    voxel_indices = jnp.stack([ii, jj], axis=-1)  # shape = (x_size, y_size, 2)
    result = voxel_indices[mask]  # shape = (N, 2)

    return np.array(result+0.5)
  
   
def voxel2legal_edge_2d(target_sdf:Callable, cube:np.ndarray,range0:np.ndarray,edge_length:float)->np.ndarray:
    cube = np.array(cube)
    sorted_cube = sort_voxels(cube)
    legal_edges = []
    start = np.array([range0[0][0],range0[1][0],range0[0][0],range0[1][0]])
    for entry in sorted_cube:
        coord = edge_length * entry + [range0[0][0],range0[1][0]]
        if(target_sdf(jnp.array(coord))>0.):
            edges = get_all_edge_for_cube(entry)
            adjacent_cubes = get_all_adjacent_cube(entry)
            for iter in range(4):
                if(find_pair_in_arr(sorted_cube,adjacent_cubes[iter]) == -1):
                    legal_edges.append(edge_length * edges[iter]+start)
    return np.array(legal_edges)


def get_cycles_from_sdf(target_sdf:Callable,range0:np.ndarray,edge_length:float,sdf_val_threshold:float,min_edge_num:int = 10) -> List[np.ndarray]:
    cubes = sdf2initial_voxels_2d(target_sdf,range0,edge_length,sdf_val_threshold)
    legal_edges = voxel2legal_edge_2d(target_sdf,cubes,range0,edge_length)
    graph = nx.Graph()
    for edge in legal_edges:
        graph.add_edge((edge[0],edge[1]),(edge[2],edge[3]))
    cycles = nx.cycle_basis(graph)
    res = []
    for cycle in cycles:
        flattened = np.array([coord for pt in cycle for coord in pt], dtype=np.float64)
        if(len(flattened)>4*min_edge_num):res.append(flattened)
    return res
  
    
   

    

            