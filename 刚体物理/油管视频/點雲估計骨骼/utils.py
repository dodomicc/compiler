import numpy as np
from skimage.util import map_array
from scipy import sparse
import  scipy.ndimage as ndi






def get_pca_ratio(A:np.ndarray, threshold: float)->bool:
    A -= np.mean(A,axis = 0)
    T = A.T @ A
    eigen_vals, _ = np.linalg.eigh(T)
    lambda_ratio = np.max(eigen_vals)/np.sum(eigen_vals)
    return lambda_ratio>threshold

def get_pca_dir(A:np.ndarray)->np.ndarray:
    A -= np.mean(A,axis = 0)
    T = A.T @ A
    eigen_vals, eigen_vecs = np.linalg.eigh(T)
    return eigen_vecs @ eigen_vals.T/np.sum(eigen_vals)

# 將一個體素的鄰接index展開
def index2pattern(index):
    voxel_bitmask = 2 ** np.arange(27)
    return (voxel_bitmask & index> 0).reshape((3,3,3))

# 將一個體素的鄰接展開壓縮
def pattern2index(pattern:np.ndarray):
    return np.sum(2 ** np.flatnonzero(pattern))

# 從體素中提取鄰接圖
def voxels2graph(voxels:np.ndarray, connectivity: int)->any:
    footprint = ndi.generate_binary_structure(voxels.ndim,connectivity)
    center =  np.array(footprint.shape)//2
    footprint_indices = np.stack(np.nonzero(footprint), axis=-1)
    offsets = footprint_indices - center
    def ravel_index(indexs:np.ndarray, voxels: np.ndarray)->np.ndarray:
        ravel_factor = voxels.shape[1:] + (1,)
        ravel_factor = np.cumprod(ravel_factor[::-1])[::-1]    
        return np.sum(indexs * ravel_factor,axis = 1)
    raveled_offsets = ravel_index(offsets, voxels)
    raveled_distance = np.linalg.norm(offsets, axis=1)
    sorted_raveled_offsets = (raveled_offsets[np.argsort(raveled_distance)])[1:]
    sorted_raveled_distance = np.sort(raveled_distance)[1:]
    nodes = np.flatnonzero(voxels)
    neighbors = nodes[:,np.newaxis] + sorted_raveled_offsets 
    neighbors_mask = voxels.reshape(-1)[neighbors]
    neighbors_num = np.sum(neighbors_mask,axis = 1)
    neighbors_from = np.repeat(nodes, neighbors_num)
    neighbors_to = neighbors[neighbors_mask]
    neighbors_from = map_array(neighbors_from, nodes, np.arange(nodes.size))
    neighbors_to = map_array(neighbors_to, nodes, np.arange(nodes.size))
    distance_full = np.broadcast_to(sorted_raveled_distance,neighbors.shape)
    distance = distance_full[neighbors_mask]
    mat = sparse.coo_matrix(
            (distance, (neighbors_from, neighbors_to)),
            shape=(nodes.size, nodes.size)
            )
    graph = mat.tocsr()
    return graph, nodes

# 計算每個前景體素的鄰接狀態以及返回展平位置
def voxels2state(voxels:np.ndarray)->any:
    i,j,k = np.mgrid[0:3,0:3,0:3]
    footprint = np.zeros((27,3),dtype=np.int32)
    footprint[:,0] = i.reshape(-1)
    footprint[:,1] = j.reshape(-1)
    footprint[:,2] = k.reshape(-1)
    center =  np.array([1,1,1])
    offsets = footprint - center
    def ravel_index(indexs:np.ndarray, voxels: np.ndarray)->np.ndarray:
        ravel_factor = voxels.shape[1:] + (1,)
        ravel_factor = np.cumprod(ravel_factor[::-1])[::-1]    
        return np.sum(indexs * ravel_factor,axis = 1)
    raveled_offsets = ravel_index(offsets, voxels)
    nodes = np.flatnonzero(voxels)
    neighbors = nodes[:,np.newaxis] + raveled_offsets
    neighbors_mask = voxels.reshape(-1)[neighbors]
    neighbors_index = neighbors_mask * (2 ** np.arange(27))
    neighbors_index = np.sum(neighbors_index, axis = 1)
    return neighbors_index, nodes