import numpy as np


# 计算两个向量减法的对于n个向量的雅可比矩阵
def xi_minus_xj_jac(i:int,j:int,n:int)->np.ndarray:
    jac = np.zeros((3,n * 3))
    for k in range(3):
        jac[k][3*i + k] += 1.
        jac[k][3*j + k] -= 1.
    return jac

# 计算两个向量加法的对于n个向量的雅可比矩阵
def xi_add_xj_jac(i:int,j:int,n:int)->np.ndarray:
    jac = np.zeros((3,n * 3))
    for k in range(3):
        jac[k][3*i + k] += 1.
        jac[k][3*j + k] += 1.
    return jac

# 计算叉乘之后的向量关于两个原向量的雅可比矩阵
def x_cross_y_jac(x:np.ndarray,y:np.ndarray)->np.ndarray:
    a1, a2, a3 = x
    b1, b2, b3 = y
    Ja = np.array([[ 0,  b3, -b2],
                [-b3, 0,   b1],
                [ b2,-b1,  0 ]], dtype=float)

    Jb = np.array([[ 0, -a3,  a2],
                [ a3,  0, -a1],
                [-a2, a1,  0 ]], dtype=float)

    return np.hstack([Ja, Jb])

# 计算一个向量的范数关于这个向量的梯度
def x_norm_jac(x:np.ndarray)->np.ndarray:
    return x/(np.linalg.norm(x) + 1e-6)

# 计算一个向量关于它的规范化向量的雅可比矩阵
def x_normalize_jac(x:np.ndarray)->np.ndarray:

    norm = np.linalg.norm(x)+1e-6
    x = x/norm
    x1 = x.reshape((1,x.shape[0]))
    x2 = x.reshape((x.shape[0],1))
    I = np.identity(x.shape[0])
    res =  (1./norm ) * (I - x2 @ x1)
    return res

# 计算两个向量点积的梯度
def x_y_dot_jac(x:np.ndarray,y:np.ndarray)->np.ndarray:
    size = x.size
    x = x.reshape((1,size))
    y = y.reshape((1,size))
    return np.concatenate([y,x],axis=1)


def x1_x2_cos_jac(x1:np.ndarray,x2:np.ndarray)->np.ndarray:
    normalized_x1_jac = x_normalize_jac(x1)
    normalized_x2_jac = x_normalize_jac(x2)
    x1_normalized_jac = np.hstack([normalized_x1_jac,np.zeros((3,3))])
    x2_normalized_jac = np.hstack([np.zeros((3,3)),normalized_x2_jac])
    x1_x2_normalized_jac = np.vstack([x1_normalized_jac,x2_normalized_jac])
    x1_x2_dot_jac = x_y_dot_jac(x1/np.linalg.norm(x1),x2/np.linalg.norm(x2))
    return x1_x2_dot_jac @ x1_x2_normalized_jac



def x1_cross_x2_x3_cross_x4_cos_jac(x1: np.ndarray,x2: np.ndarray,x3:np.ndarray,x4:np.ndarray)->np.ndarray:
    x1_cross_x2_jac = x_cross_y_jac(x1,x2)
    x3_cross_x4_jac = x_cross_y_jac(x3,x4)
    res_1 = np.hstack([x1_cross_x2_jac,np.zeros((3,6))])
    res_2 = np.hstack([np.zeros((3,6)),x3_cross_x4_jac])
    res = np.vstack([res_1,res_2])
    cross1 = np.cross(x1,x2)
    cross2 = np.cross(x3,x4)
    res =  x1_x2_cos_jac(cross1,cross2) @  res
    return res

def normals_cos_jac(x1: np.ndarray,x2: np.ndarray,x3:np.ndarray,x4:np.ndarray)->np.ndarray:
    x1_minus_x2 = x1 - x2
    x1_minus_x3 = x1 - x3
    x4_minus_x2 = x4 - x2
    x4_minus_x3 = x4 - x3
    x_diff_matrix = np.vstack([xi_minus_xj_jac(0,1,4),
                                xi_minus_xj_jac(0,2,4),
                                xi_minus_xj_jac(3,1,4),
                                xi_minus_xj_jac(3,2,4)])
    return x1_cross_x2_x3_cross_x4_cos_jac(x1_minus_x2,x1_minus_x3,x4_minus_x2,x4_minus_x3 ) @ x_diff_matrix

   
    
def huber_abs(x:float,delta:float)->float:
    if(np.abs(x)>delta): return abs(x) - delta
    return (x ** 2)/(2 * delta)


def huber_deritive(x:float,delta:float)->float:
    if(np.abs(x)>delta): return np.sign(x)
    return x/delta


def  point_to_plane_dist_jac(x:np.ndarray,tri_1:np.ndarray,tri_2:np.ndarray,tri_3:np.ndarray)->np.ndarray:
    # 输入的前三个分量是一个点，后面三个点是一个平面，计算该点到这个平面的Huber距离关于这十二个点的梯度
    tri_nor = np.cross(tri_2-tri_1,tri_2-tri_3)/np.linalg.norm(np.cross(tri_2-tri_1,tri_2-tri_3))
    xy = x - tri_1
    xy_dot_nor_jac = x_y_dot_jac(xy,tri_nor)
    x_minus_y_jac = xi_minus_xj_jac(0,1,4)
    zy_cross_zw_normalize_jac =  x_normalize_jac(np.cross(tri_2-tri_1,tri_2-tri_3)) @ x_cross_y_jac(tri_2-tri_1,tri_2-tri_3) @ np.vstack([xi_minus_xj_jac(1,0,3),xi_minus_xj_jac(1,2,3)])
    zy_cross_zw_normalize_jac = np.hstack([np.zeros((3,3)), zy_cross_zw_normalize_jac])
    dot_jac = xy_dot_nor_jac @ np.vstack([x_minus_y_jac,zy_cross_zw_normalize_jac])
    res = dot_jac * huber_deritive(np.dot(xy,tri_nor),0.0001)
    return res

def points_dist_constraint_jac(x:np.ndarray,y:np.ndarray)->np.ndarray:
    x_minus_y_jac = xi_minus_xj_jac(0,1,2)
    xy_norm_jac = x_norm_jac(x-y)
    return xy_norm_jac @ x_minus_y_jac

def tri_area_jac(A:np.ndarray,B:np.ndarray,C:np.ndarray)->np.ndarray:
    AC = A - C
    AB = A - B
    ac_jac = xi_minus_xj_jac(0,2,3)
    ab_jac = xi_minus_xj_jac(0,1,3)
    jac_1 = np.vstack([ac_jac,ab_jac])
    jac_2 = x_cross_y_jac(AC,AB)
    jac_3 = x_norm_jac(np.cross(AC,AB))
    return 0.5 * jac_3 @ jac_2 @ jac_1

def volume_tri_part_jac(A:np.ndarray,B:np.ndarray,C:np.ndarray)->np.ndarray:
    AC = A - C
    AB = A - B
    ac_jac = xi_minus_xj_jac(0,2,3)
    ab_jac = xi_minus_xj_jac(0,1,3)
    jac_1 = np.vstack([ac_jac,ab_jac])
    jac_2 = x_cross_y_jac(AC,AB)
    jac_3 = x_normalize_jac(np.cross(AC,AB))
    jac_4 = jac_3 @ jac_2 @ jac_1
    A_add_B_add_C_jac = np.hstack([np.identity(3),np.identity(3),np.identity(3)])
    A_add_B_add_C = A + B + C
    AC_AB_cross_normalize = np.cross(AC,AB)/(np.linalg.norm(np.cross(AC,AB)) + 1e-6)
    res = x_y_dot_jac(A_add_B_add_C,AC_AB_cross_normalize) @ np.vstack([A_add_B_add_C_jac,jac_4])
    res = 2  * np.dot(A_add_B_add_C,AC_AB_cross_normalize) * tri_area_jac(A,B,C) +  np.linalg.norm(np.cross(AC,AB)) * res
    return (1/6) * res


