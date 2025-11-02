import numpy as np
from typing import *
def row_swap_qr(matrix: np.ndarray, i: int,  q_accum: np.ndarray):
    col_segment = matrix[i:, i]
    max_index = np.argmax(np.abs(col_segment))
    target_row = i + max_index
    q_accum[[i,target_row]] = q_accum[[target_row,i]]
    matrix[[i,target_row]] = matrix[[target_row,i]]

    
    
    
def householder_qr(matrix: np.ndarray, i: int, q_accum: np.ndarray):
    col_segment = matrix[i:, i]
    if(abs(col_segment[0])<1e-5): return
    e1 = np.zeros_like(col_segment)
    e1.fill(0)
    e1[0] = 1.
    v = col_segment + np.sign(col_segment[0])* np.linalg.norm(col_segment) * e1
    norm_v = np.linalg.norm(v)
    v = v / norm_v
    H =  - 2. * np.outer(v,v)
    matrix[i:,:] =matrix[i:,:] +  H @ matrix[i:,:]
    q_accum[i:,:] =q_accum[i:,:] +  H @ q_accum[i:,:]
 
    
    
def qr(mat0: np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    mat = mat0.copy()
    size = mat.shape[0]
    q_accum = np.identity(size)
    for i in range(size):
        row_swap_qr(mat,i,q_accum)
        householder_qr(mat,i,q_accum)
    return q_accum.transpose(), mat



def max_offdiag_abs_index_jacobi_diag(mat_src: np.ndarray, abs_buffer: np.ndarray):
    mat_abs: np
    mat_abs = abs_buffer
    np.abs(mat_src, out=abs_buffer)
    np.fill_diagonal(mat_abs, 0)
    idx = np.argmax(mat_abs)
    row = idx // mat_abs.shape[0]
    col = idx % mat_abs.shape[0]
    return row, col


def givens_jacobi_diag(a11: float, a22: float, a12: float) -> np.ndarray:
    delta = a11 - a22
    beta = 2.0 * a12
    t = beta / (delta + (delta**2 + beta**2) ** 0.5)
    cos = 1.0 / (1 + t**2) ** 0.5
    sin = t / (1 + t**2) ** 0.5
    orth = np.zeros((2, 2))
    orth[0, 0] = cos
    orth[0, 1] = -sin
    orth[1, 0] = sin
    orth[1, 1] = cos
    return orth


def jacobi_diag(mat0:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    mat = mat0.copy()
    size = mat.shape[0]
    orth_total = np.identity(size)
    abs_buffer = np.identity(size)
    row_min = np.zeros_like(orth_total[0,:])
    row_max = np.zeros_like(orth_total[0,:]) 
    col_min = np.zeros_like(orth_total[:,0]) 
    col_max = np.zeros_like(orth_total[:,0]) 
    for iter in range(100000):
        height,width = max_offdiag_abs_index_jacobi_diag(mat,abs_buffer)
        print(f"{iter} ----- {abs(mat[height][width])}")
        if(height == width or abs(mat[height][width])<=1e-8): break
        min_idx = min(height,width)
        max_idx = max(height,width)
        orth_mat = givens_jacobi_diag(mat[min_idx][min_idx],mat[max_idx][max_idx],mat[min_idx][max_idx])
        # np.dot(orth_prod,orth_mat,out=orth_prod)
        col_min[:] = orth_total[:,min_idx]
        col_max[:] = orth_total[:,max_idx]
  
        orth_total[:,min_idx] = orth_mat[0][0] * col_min + orth_mat[1][0] * col_max
        orth_total[:,max_idx] = orth_mat[0][1] * col_min + orth_mat[1][1] * col_max
        # for i in range(size): transpose_temp[i,:] = orth_mat[:,i]
        orth_mat_transpose = orth_mat.transpose()
        # np.dot(transpose_temp,mat,out = mat_temp)
        row_min[:] = mat[min_idx,:]
        row_max[:] = mat[max_idx,:]
        mat[min_idx,:] =  orth_mat_transpose[0][0] * row_min + orth_mat_transpose[0][1] * row_max
        mat[max_idx,:] =  orth_mat_transpose[1][0] * row_min + orth_mat_transpose[1][1] * row_max
        # np.dot(mat_temp,orth_mat,out = mat)
        col_min[:] = mat[:,min_idx]
        col_max[:] = mat[:,max_idx]
        mat[:,min_idx] =  orth_mat[0][0] * col_min + orth_mat[1][0] * col_max
        mat[:,max_idx] =  orth_mat[0][1] * col_min + orth_mat[1][1] * col_max
    return mat,orth_total




    

def solve_linear_system(A: np.ndarray,b:np.ndarray)->np.ndarray:  
    Q, R = qr(A.T @ A)
    Qtb = Q.T @ A.T @ b
    n = R.shape[1]
    x = np.zeros((n, 1))
    x[n-1] = 0.0 if abs(R[n-1, n-1]) < 1e-6 else Qtb[n-1] / R[n-1, n-1]
    for i in reversed(range(n-1)): x[i] = 0.0 if abs(R[i, i]) < 1e-6 else (Qtb[i] - R[i, i+1:] @ x[i+1:]) / R[i, i] 
    return x