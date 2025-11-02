import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def point_to_segment_distance(coords: np.ndarray) -> np.ndarray:
    a = coords[0]
    b = coords[-1]
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    ap = coords - a
    t = np.clip((ap @ ab) / ab_len2, 0, 1)
    proj = a + np.outer(t, ab)
    dists = np.linalg.norm(coords - proj, axis=1, keepdims=True)
    return dists

def split_segments(coords: np.ndarray, pca_threshold: float = 0.99) -> list:
    len0 = np.linalg.norm(coords[-1] - coords[0])
    if len0 < 0.001 or coords.shape[0] < 3:
        return [np.array([coords[0], coords[-1]])]

    center = np.mean(coords, axis=0)
    centered = coords - center

    A = centered.T @ centered
    eigen_val, _ = np.linalg.eigh(A)
    main_dir = (coords[-1] - coords[0])
    main_dir = main_dir / np.linalg.norm(main_dir)

    pca_ratio = np.sum((centered @ main_dir) ** 2) / np.sum(eigen_val)
    dists = point_to_segment_distance(coords)

    if pca_ratio < pca_threshold:
        idx = np.argmax(dists)
        if 0 < idx < coords.shape[0] - 1:
            return [coords[:idx+1], coords[idx:]]
    
    return [np.array([coords[0], coords[-1]])]

def recursive_split_segments(coords: np.ndarray, threshold: float) -> np.ndarray:

    if coords.shape[0] == 1:
        return np.array([coords[0], coords[0]])

    segments = [coords]
    final_segments = []

    while segments:
        segment = segments.pop()
        new_segments = split_segments(segment, threshold)
        if len(new_segments) == 2:
            segments.extend(new_segments)
        else:
            final_segments.append(new_segments[0])

    # 建構圖：節點為起止點，邊為 segment
    nodes = {}
    node_to_idx = {}
    G = nx.Graph()

    for seg in final_segments:
        pt1, pt2 = tuple(seg[0]), tuple(seg[1])
        for pt in (pt1, pt2):
            if pt not in node_to_idx:
                idx = len(nodes)
                node_to_idx[pt] = idx
                nodes[idx] = np.array(pt)
        i, j = node_to_idx[pt1], node_to_idx[pt2]
        G.add_edge(i, j)

    # 從起始點延圖遍歷回復節點順序
    start_idx = node_to_idx[tuple(coords[0])]
    visited = {start_idx}
    path = [nodes[start_idx]]
    neighbors = list(G.neighbors(start_idx))

    if not neighbors:
        return np.array(path)

    current = neighbors[0]
    while current not in visited:
        path.append(nodes[current])
        visited.add(current)
        next_neighbors = [n for n in G.neighbors(current) if n not in visited]
        if not next_neighbors:
            break
        current = next_neighbors[0]

    return np.array(path)



class SegmentKDTree:
    def __init__(self, segments: np.ndarray):
        """
        segments: (N, 2) array of N 2D segments. Each segment has two endpoints.
        """
        self.segments = segments
        start = segments[:segments.shape[0]-1,:]
        end = segments[1:,:]
        centers = (start + end)/2
        endpoints = np.zeros((3 * (segments.shape[0] - 1),segments.shape[1]))
        endpoints[np.arange(endpoints.shape[0])%3 == 0,:] = start
        endpoints[np.arange(endpoints.shape[0])%3 == 1,:] = centers
        endpoints[np.arange(endpoints.shape[0])%3 == 2,:] = end
        # 计算每个线段的起点，中点和终点作为 KDTree 索引
        # centers = segments.mean(axis=0)  # (N, 2)
        self.kdtree = cKDTree(endpoints)

    def query_point(self, point: np.ndarray, k: int = 8):
        """
        查询最近点到路径上的最近投影点
        - point: (2,) query point
        - k: 考虑最近的 k 条线段（更快 + 更近似）
        """
        # 先找最近的 k 个线段中心
        _, indices = self.kdtree.query(point, k=k)
        closest_proj = None
        min_dist = float('inf')

        for idx in np.atleast_1d(indices):
            a = self.segments[(idx//3)]
            b = self.segments[(idx//3) + 1] 
            ap = point - a
            ab = b - a
            t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
            proj = a + t * ab
            dist = np.linalg.norm(point - proj)
            if dist < min_dist:
                min_dist = dist
                closest_proj = proj

        return closest_proj, min_dist