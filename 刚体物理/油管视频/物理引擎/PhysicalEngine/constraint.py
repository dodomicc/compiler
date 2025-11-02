import jax.numpy as jnp
import jax.nn as jnn
from jax import *
from typing import *


def construct_updater(f_func: Callable, grad_func: Callable) -> Callable:
    def updater(x: jnp.ndarray) -> jnp.ndarray:
        f_val = f_func(x)
        grad_val = grad_func(x)
        return -f_val / (jnp.linalg.norm(grad_val)**2 + 1e-12) * grad_val
    return updater

def smin(d1:float, d2:float, k:float)->float:
    h = jnp.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d1 * h + d2 * (1 - h) - k * h * (1 - h)

def smax(d1:float, d2:float, k:float)->float:
    return -smin(-d1,-d2,k)

def sdbox(p: jnp.ndarray, size: jnp.ndarray) -> float:
    q = jnp.abs(p)
    outside_dist = jnp.linalg.norm(jnp.maximum(q, 0.0))
    inside_dist = jnp.minimum(jnp.maximum(q[0], q[1]), 0.0)
    return outside_dist + inside_dist

def sd_rounded_box(p:jnp.array, size:jnp.array, r:float)->float:
    d = jnp.abs(p) - size + r
    out = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1) - r
    out += jnp.minimum(jnp.maximum(d[...,0], d[...,1]), 0.0)
    return out

def remap(x1:float,x2:float,y1:float,y2:float,x:float)->float:
    h = (x-x1)/(x2-x1 + 1e-6)
    return y1 + (y2-y1) * h
    

def angle_constraint_2d(points: jnp.ndarray, penalty_scale:float) -> float:
    dirs = jnp.array([
        points[2:4] - points[0:2],
        points[4:6] - points[2:4]
    ])
    a, b = dirs[0], dirs[1]
    cos_theta = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)
    loss = - penalty_scale * jnp.log((1.+cos_theta)/2.+ 1e-8)
    return loss

def edge_length_constraint_2d(points:jnp.array, length:float ,penalty_scale:float) ->float:
    points = points.reshape((-1,2))
    edge = points[1] - points[0]
    return penalty_scale * ((jnp.linalg.norm(edge)-length)**2)

def smooth_loop_angle_constraint_2d(points: jnp.ndarray, penalty_scale:float) -> float:
    arr_reshape = points.reshape((-1,2))
    arr2 = jnp.hstack([jnp.roll(arr_reshape,axis=0,shift=1),arr_reshape,jnp.roll(arr_reshape,axis = 0, shift=-1)])
    loss = jnp.sum(vmap(angle_constraint_2d,(0,None))(arr2,penalty_scale))
    return loss

def smooth_loop_length_constraint_2d(points: jnp.ndarray, length: float, penalty_scale:float) -> float:
    arr_reshape = points.reshape((-1,2))
    arr2 = jnp.hstack([arr_reshape,jnp.roll(arr_reshape,axis = 0, shift=-1)])
    loss = jnp.sum(vmap(edge_length_constraint_2d,(0,None,None))(arr2,length,penalty_scale))
    return loss

def point_to_segment_distance(P, A, B):
    AB = B - A
    AP = P - A
    t = jnp.clip(jnp.dot(AP, AB) / (jnp.dot(AB, AB) + 1e-8), 0.0, 1.0)
    closest = A + t * AB
    return jnp.linalg.norm(P - closest)

def segment_to_segment_distance(A, B, C, D):
    d1 = point_to_segment_distance(A, C, D)
    d2 = point_to_segment_distance(B, C, D)
    d3 = point_to_segment_distance(C, A, B)
    d4 = point_to_segment_distance(D, A, B)
    return jnp.min(jnp.array([d1, d2, d3, d4]))

def self_ballon_constraint_2d(points: jnp.ndarray,expansion_decay_rate:float,expansion_force_scale:float):


    points = points.reshape(-1,2)
    cen = jnp.array([jnp.mean(points[:,0]), jnp.mean(points[:,1])])
    diff = points - cen
    dist = vmap(jnp.linalg.norm)(diff)
    dist *= jnp.exp(-expansion_decay_rate * dist)
    dist_sum = jnp.sum(dist)
    loss = -expansion_force_scale * dist_sum
    return loss



def step0(point: jnp.array, updater:Callable,max_iter_times:int=300):
    for i in range(max_iter_times):
        new_point = updater(point,i,max_iter_times)
  
        if(isinstance(new_point,bool)  and new_point== False):
            break
        else:
            point = new_point

    return point
        
    
    
