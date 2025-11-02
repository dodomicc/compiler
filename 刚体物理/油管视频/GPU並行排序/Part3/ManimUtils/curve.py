import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from utils import *
from typing import *


def get_perpendicular_control_point(p0, p1, scale=0.25):
    """
    给定两个点，返回一个控制点，位于垂直平分线上，距离为原线段长度的 scale 倍。
    :param p0: 起点（np.array([x, y, z])）
    :param p1: 终点
    :param scale: 控制点距离长度比例，默认为 1/4
    :return: 控制点坐标（np.array）
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    midpoint = (p0 + p1) / 2
    direction = p1 - p0

    # 取平面内二维方向（忽略z），计算其垂直向量
    dx, dy = direction[:2]
    perp = np.array([-dy, dx])  # 垂直向量

    # 单位化
    perp = perp / np.linalg.norm(perp + 1e-8)  # 防止除以0

    # 扩展为三维向量
    perp_3d = np.array([perp[0], perp[1], 0.0])

    # 控制点位置 = 中点 + 垂直方向 * 距离
    length = np.linalg.norm(direction) * scale
    control_point = midpoint + perp_3d * length

    return control_point
    
def quadratic_bezier_points(p0, p1, num_points=100, scale=0.25):
    """
    根据两点p0, p1，生成100个在二阶贝塞尔曲线上的点
    :return: (100, 3) 的 NumPy 数组
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    c = get_perpendicular_control_point(p0, p1, scale)

    t = np.linspace(0, 1, num_points).reshape(-1, 1)  # shape: (100, 1)
    points = (1 - t)**2 * p0 + 2 * (1 - t) * t * c + t**2 * p1  # 二阶贝塞尔公式
    return points

def get_curve(p0:np.ndarray, p1:np.ndarray, color:Color=BLUE,num_points :float = 100, scale:float= 0.25)->VMobject:
     curve = VMobject(color)
     curve.set_points_as_corners(quadratic_bezier_points(p0,p1,num_points,scale))
     return curve
     
    

