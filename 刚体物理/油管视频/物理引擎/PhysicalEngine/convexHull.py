import jax.numpy as jnp
import jax.nn as jnn
from jax import *
from typing import *
import numpy as np 
from scipy.spatial import ConvexHull


def get_converHull(points:np.ndarray):
    return ConvexHull(points)


def get_converHull_vertices(hull:ConvexHull):
    return hull.points[hull.vertices]


def get_converHull_equations(hull:ConvexHull):
    return hull.equations


def points_to_segments(points):
    """
    将一系列点 (M+1, 2) 转换为线段数组 (M, 4)
    每条线段是 [x0, y0, x1, y1]
    """
    p0 = points[:-1]  # (M, 2)
    p1 = points[1:]   # (M, 2)
    segments = np.hstack([p0, p1])  # (M, 4)
    return segments


def points_to_segments_min_distance(x, hull):
    """
    输入:
        x: (N, 2) 点集
        segments: (M, 4)，每行是 (x0, y0, x1, y1) 表示一条线段
    输出:
        dists: (N,) 每个点到所有线段中最近一条线段的距离
    """
    segments = points_to_segments(get_converHull_vertices(hull))
    N = x.shape[0]
    M = segments.shape[0]

    # 拆出线段端点 A, B
    A = segments[:, 0:2]  # (M, 2)
    B = segments[:, 2:4]  # (M, 2)
    AB = B - A            # (M, 2)
    AB_len2 = np.sum(AB**2, axis=1) + 1e-8  # 避免除以0

    dists = np.zeros(N)

    for i in range(N):
        P = x[i]                   # (2,)
        AP = P - A                # (M, 2)
        t = np.sum(AP * AB, axis=1) / AB_len2   # (M,)
        t = np.clip(t, 0.0, 1.0)[:, np.newaxis] # (M, 1)
        proj = A + t * AB                         # 最近点 (M, 2)
        dist = np.linalg.norm(P - proj, axis=1)   # 距离 (M,)
        dists[i] = np.min(dist)                   # 最小距离

    return dists


def is_inside_hull(points: np.ndarray,hull: ConvexHull)->np.ndarray:
    edges = hull.equations
    is_inside = np.dot(np.hstack([points, np.ones((points.shape[0], 1))]),edges.transpose())
    is_inside = np.max(is_inside,axis=1)
    is_inside = np.where(is_inside<1e-6,-1,1)
    return is_inside

def sdf_convex_hull_points(points: np.ndarray,hull: ConvexHull)->np.ndarray:
    return is_inside_hull(points,hull) * points_to_segments_min_distance(points,hull)

def sdf_convex_hull_point(point: np.ndarray,hull: ConvexHull)->np.ndarray:
    points = np.array([
        point
    ])
    return sdf_convex_hull_points(points,hull)[0]

