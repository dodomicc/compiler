from numpy.linalg import *
from numba import njit
from scipy.spatial import cKDTree
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import *

@njit
def compute_normals(points, indices):
    N = points.shape[0]
    normals = np.zeros((N, 3))
    for i in range(N):
        A = points[indices[i,:]]
        mean = np.sum(A, axis=0)/A.shape[0]
        A = A - mean
        cov = A.T @ A
        _, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]
    return normals

@njit
def align_normals_mst(corrected_normals, mst_indices, mst_indptr, root_index=0):
    N = corrected_normals.shape[0]
    visited = np.zeros(N, dtype=np.bool_)
    queue = [root_index]
    visited[root_index] = True
    while len(queue) > 0:
        u = queue.pop(0)  # Numba 支持 list 作 queue，但不支持 deque
    
        for i in range(mst_indptr[u], mst_indptr[u + 1]):
            v = mst_indices[i]
            if not visited[v]:
         
                if np.dot(corrected_normals[u], corrected_normals[v]) < 0:
          
                    corrected_normals[v] = -corrected_normals[v]
                visited[v] = True
                queue.append(v)
    # print(np.sum(visited))


def point_cloud2cens_nors(vertices:np.ndarray, max_neighbors:int = 64, radius:float = 0.1)->any:
    tree = cKDTree(vertices)
    distances, indices = tree.query(vertices, max_neighbors)
    normals = np.zeros((vertices.shape[0], 3))
    normals = compute_normals(vertices,indices)

    # 根據平均點和法線重新估計
    distances, indices = tree.query(vertices, max_neighbors)
    indices_mask = (indices != np.arange(vertices.shape[0])[:,None]) & (distances<radius)
    neighbors_num = np.sum(indices_mask, axis = 1)

    neighbors_from = np.repeat(np.arange(vertices.shape[0]),neighbors_num)
    neighbors_to = indices[indices_mask]
    nors1 = normals[neighbors_from]
    nors2 = normals[neighbors_to]
    cos_nors1_nors2 = np.abs(np.sum(nors1 * nors2, axis = 1))
    dist  = 1. - cos_nors1_nors2
    
    
    mat = sparse.coo_matrix((dist,(neighbors_from,neighbors_to)),shape=(normals.shape[0],normals.shape[0]))
    csr_graph = mat.tocsr()
    csr_graph = csr_graph + csr_graph.T
    mst = minimum_spanning_tree(csr_graph)
    mst = mst + mst.T
 
    mst  = mst.tocsr()
    mst_indptr = mst.indptr
    mst_indices = mst.indices
    corrected_normals = normals
    n, labels = connected_components(csgraph=mst, directed=False)
    # print(n)
    counts = np.bincount(labels)
    largest_component_label = np.argmax(counts)
    largest_component_indices = np.where(labels == largest_component_label)[0]
    align_normals_mst(corrected_normals,mst_indices,mst_indptr, largest_component_indices[0])

    return vertices[largest_component_indices],corrected_normals[largest_component_indices]