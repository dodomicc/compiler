
import jax.numpy as jnp
import jax.nn as jnn
from jax import *
from typing import *
import numpy as np 


def points2loop_np(point:np.ndarray):
    res = np.concatenate([point,[point[0]]])
    return res

def points2loop_jnp(point:jnp.array):
    res = jnp.concatenate([point,[point[0]]])
    return res

def add_ones_column_right(arr:np.ndarray)->np.ndarray:
    return np.hstack([arr,np.ones((arr.shape[0],1))])

def add_ones_column_right_jax(arr:jnp.ndarray)->jnp.ndarray:
    return jnp.hstack([arr,jnp.ones((arr.shape[0],1))])