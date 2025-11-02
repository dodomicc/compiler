import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from ManimUtils.utils import *

from typing import *
import numpy as np
from ManimUtils.basic_2d_scenes import *
from ManimUtils.glsl_render import *
from pathlib import *
from concurrent.futures import ThreadPoolExecutor
from shader_exectuer import * 
from pymorton import *

LINEARVELOCITY = 1
LINEARVELOCITY_LENGTH = 3
LINEARVELOCITY_START = "LINEARVELOCITY_START"

ANGULARVELOCITY = 2
ANGULARVELOCITY_LENGTH = 3
ANGULARVELOCITY_START = "ANGULARVELOCITY_START"

POS = 3
POS_LENGTH = 3
POS_START = "POS_START"



MASS = 4
MASS_LENGTH = 1
MASS_START = "MASS_START"


SINGLE_PARTICLE_LENGTH = "SINGLE_PARTICLE_LENGTH"
TOTAL_PARTICLE_NUM = "TOTAL_PARTICLE_NUM"

def compute_morton(nums:list[int,int,int])->int:
    return interleave3(nums[0],nums[1],nums[2])

def get_arr_from_img(img:Image.Image)->np.ndarray:
    return np.array(img)

def get_data_from_img(arr:np.ndarray,idx:int)-> float:
    row = int(np.floor(idx/128))
    col = int(idx%128)
    val = decode_float(tuple(arr[row,col]))
    return val


def set_data_to_img(arr:np.ndarray,idx:int,num:float):
    width = 128
    row = int(np.floor(idx/width))
    col = int(idx%width)
    arr[row, col] = list(encode_float(num)) 
    

def encode_types(types:List[int])->float:
    res = np.zeros(4)
    while(len(types)<8):
        types.append(0)
    for i,type in enumerate(types):
        idx = int(np.floor(i/2))
        interior_idx = i%2
        res[idx]+=type * (1 + 15 * (1 - interior_idx))
    return decode_float(tuple(res.astype(int)))


def decode_types(num:float)->List[int]:
    res :List[float]= []
    pixel = encode_float(num)
    for i , entry in enumerate(pixel):
        res.append(int(np.floor(entry/16)))
        res.append(int(entry%16))
    return res

def get_total_img_length(types:List[int],particle_num:int)->int:
    total_length = 2
    single_length = 0
    for i, type in enumerate(types):
        if(type == LINEARVELOCITY):
            single_length+=LINEARVELOCITY_LENGTH
        if(type == ANGULARVELOCITY):
            single_length+= ANGULARVELOCITY_LENGTH
        if(type == POS):
            single_length+= POS_LENGTH
        if(type == MASS):
            single_length+= MASS_LENGTH
    total_length += single_length * particle_num
    return total_length

def get_img_by_length(length: int)->Tuple[np.ndarray,Image.Image]:
    return generate_standard_image(length)

def set_metadata(arr: np.ndarray,types:List[int],particle_num:int):
    set_data_to_img(arr,0,float(particle_num))
    set_data_to_img(arr,1,encode_types(types))
 

def get_particles_state(arr: np.ndarray)->dict:
    total_particle_num = get_data_from_img(arr,0)
    types = decode_types(get_data_from_img(arr,1))
    curIdx = 0
    res = {}
    for i, type in enumerate(types):
        if(type == LINEARVELOCITY):
            res[LINEARVELOCITY_START] = curIdx
            curIdx+=LINEARVELOCITY_LENGTH
        
        if(type == ANGULARVELOCITY):
            res[ANGULARVELOCITY_START] = curIdx
            curIdx+= ANGULARVELOCITY_LENGTH
            
        if(type == POS):
            res[POS_START] = curIdx
            curIdx+= POS_LENGTH
            
        if(type == MASS):
            res[MASS_START] = curIdx
            curIdx+= MASS_LENGTH
    
    res[SINGLE_PARTICLE_LENGTH] = curIdx
    res[TOTAL_PARTICLE_NUM] = int(np.round(total_particle_num))
    return res



def set_float_field(arr: np.ndarray,start_id:int,particle_id:int, state:dict,num:float):
    id = 2 + state[SINGLE_PARTICLE_LENGTH] * particle_id + start_id
    set_data_to_img(arr,id,num)
    
def set_vec_field(arr: np.ndarray,start_id:int,particle_id:int, state:dict,nums:List[float]):
    id = 2 + state[SINGLE_PARTICLE_LENGTH] * particle_id + start_id
    for i,entry in enumerate(nums):
        set_data_to_img(arr,id + i,entry)
        
def get_float_field(arr: np.ndarray,start_id:int,particle_id:int, state:dict)->float:
    id =  2 + state[SINGLE_PARTICLE_LENGTH] * particle_id + start_id
    return get_data_from_img(arr,id)

def get_all_float_field(arr: np.ndarray,start_id:int, state:dict)->np.ndarray:
    res = np.zeros(state[TOTAL_PARTICLE_NUM])
    for i in range(state[TOTAL_PARTICLE_NUM]):
        res[i] = get_float_field(arr,start_id,i,state)
    return res

def get_vec2_field(arr: np.ndarray,start_id:int,particle_id:int, state:dict)->List[float]:
    id =  2 + state[SINGLE_PARTICLE_LENGTH] * particle_id + start_id
    return [
        get_data_from_img(arr,id),
        get_data_from_img(arr,id+1)
        ]
    
def get_all_vec2_field(arr: np.ndarray,start_id:int, state:dict)->np.ndarray:
    res = np.zeros((state[TOTAL_PARTICLE_NUM],2))
    for i in range(state[TOTAL_PARTICLE_NUM]):
        res[i] = get_vec2_field(arr,start_id,i,state)
    return res
    
def get_vec3_field(arr: np.ndarray,start_id:int,particle_id:int, state:dict)->List[float]:
    id =  2 + state[SINGLE_PARTICLE_LENGTH] * particle_id + start_id
    return [
        get_data_from_img(arr,id),
        get_data_from_img(arr,id+1),
        get_data_from_img(arr,id+2)
        ]
    
def get_all_vec3_field(arr: np.ndarray,start_id:int, state:dict)->np.ndarray:
    res = np.zeros((state[TOTAL_PARTICLE_NUM],3))
    for i in range(state[TOTAL_PARTICLE_NUM]):
        res[i] = get_vec3_field(arr,start_id,i,state)
    return res
    

        

        




