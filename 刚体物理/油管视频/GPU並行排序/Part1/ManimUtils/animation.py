import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from utils import *
from typing import *
import numpy as np

def smoothstep(x: float) -> float:
    t = max(0.0, min(1.0, x))
    return t * t * (3 - 2 * t)

def get_objects_arrow(
    source_obj:Mobject,
    dist_obj:Mobject,
    alpha:float = 0.2
)->Arrow:
    start_coord = source_obj.get_center()
    end_coord = dist_obj.get_center()
    start_coord = (1-alpha) * start_coord + alpha * end_coord 
    end_coord = (1-alpha) * end_coord + alpha * start_coord
    return Arrow(start_coord,end_coord)

def keep_obj_flash(obj:Mobject,start_time:float,end_time:float,flash_start_time:float,flash_end_time:float,scene:Scene):
    m0 = Group()
    m0._obj = obj 
    m0._is_added = False
    scene.add(m0)
    def m0_updater(mob,t):
        if t < start_time:
            if mob._is_added == True:
                scene.remove(mob._obj)
                mob._is_added = False
        elif t< flash_start_time:
            if mob._is_added == False:
                scene.add(mob._obj)
                mob._is_added = True
        elif t<flash_end_time:
            flag = True
            if np.abs(np.sin(7 * t))<0.5:
                flag = False
            if flag == True:
                if mob._is_added == False:
                    scene.add(mob._obj)
                    mob._is_added = True
            else:
                if mob._is_added == True:
                    scene.remove(mob._obj)
                    mob._is_added = False
        elif t<end_time:
            if mob._is_added == False:
                    scene.add(mob._obj)
                    mob._is_added = True
        else:
            if mob._is_added == True:
                scene.remove(mob._obj)
                mob._is_added = False
    keep_obj_alive_update(m0,start_time,end_time+1,scene,m0_updater)
    
def keep_obj_fade(obj:Mobject,start_time:float,end_time:float,in_end_time:float,out_start_time:float,scene:Scene):
    m0 = Group()
    m0._obj = obj 
    m0._is_added = False
    scene.add(m0)
    def m0_updater(mob,t):
        if t < start_time:
            if mob._is_added == True:
                scene.remove(mob._obj)
                mob._is_added = False
        elif t< in_end_time:
            if mob._is_added == False:
                scene.add(mob._obj)
                mob._is_added = False
            alpha = remap(start_time-0.01,in_end_time+0.01,0,1,t)
            alpha = smoothstep(alpha)
            mob._obj.set_opacity(alpha)
        elif t<out_start_time:
            mob._obj.set_opacity(1)
        elif t<end_time:
            
            alpha = remap(out_start_time-0.01,end_time+0.01,1,0,t)
            alpha = smoothstep(alpha)
            mob._obj.set_opacity(alpha)
        else:
            if mob._is_added == True:
                scene.remove(mob._obj)
                mob._is_added = False
    keep_obj_alive_update(m0,start_time,end_time+0.2,scene,m0_updater)
    
def keep_obj_fade_in(obj:Mobject,start_time:float,in_end_time:float,scene:Scene):
    m0 = Group()
    m0._obj = obj 
    m0._is_added = False
    scene.add(m0)
    def m0_updater(mob,t):
        if t < start_time:
            if mob._is_added == True:
                scene.remove(mob._obj)
                mob._is_added = False
        if t< in_end_time and t>start_time:
            if mob._is_added == False:
                scene.add(mob._obj)
                mob._is_added = True
            alpha = remap(start_time,in_end_time,0,1,t)
            alpha = smoothstep(alpha)
            mob._obj.set_opacity(alpha)
       
    keep_obj_alive_update(m0,start_time,in_end_time+10,scene,m0_updater)
    
def keep_obj_move(
    source_obj:Mobject,
    dist_obj:Mobject,
    start_time:float,
    end_time:float,
    start_move_time:float,
    end_move_time:float,
    scene:Scene
    ):
    move_obj = deepcopy(source_obj)
    def  move_obj_updater(mob,t):
        alpha = remap(start_move_time,end_move_time,0,1,t)
        alpha = smoothstep(alpha)
        cen1 = source_obj.get_center()
        cen2 = dist_obj.get_center()
        width1 = source_obj.get_width()
        width2 = dist_obj.get_width()
        cur_cen = cen1 * (1 - alpha) + cen2 * alpha
        cur_width = width1 * (1-alpha) + width2 * alpha
        mob.move_to(cur_cen)
        mob.scale(cur_width/mob.get_width())
    keep_obj_alive_update(move_obj,start_time,end_time,scene,move_obj_updater)
    
def keep_obj_move2(
    source:Mobject,
    dist:Mobject,
    target:Mobject,
    start_time:float,
    end_time:float,
    start_move_time:float,
    end_move_time:float,
    scene:Scene
    ):
  
    def  move_obj_updater(mob,t):
        alpha = remap(start_move_time,end_move_time,0,1,t)
        alpha = smoothstep(alpha)
        cen1 = source.get_center()
        cen2 = dist.get_center()
        cur_cen = cen1 * (1 - alpha) + cen2 * alpha
        mob.move_to(cur_cen)
    keep_obj_alive_update(target,start_time,end_time,scene,move_obj_updater)
    
    
    
