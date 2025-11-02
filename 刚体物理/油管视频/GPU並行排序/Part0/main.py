import os
import sys
current_path = os.path.abspath(__file__)
formula_render_path = os.path.dirname(current_path)  # 回到“公式渲染”
print(formula_render_path)
sys.path.append(formula_render_path)
from typing import *
from manimlib import *
import numpy as np
from ManimUtils.utils import *
from ManimUtils.matrix import *
from ManimUtils.prob import *
from ManimUtils.text import *
from ManimUtils.nn import *
from ManimUtils.formula import *
from ManimUtils.animation import *
from itertools import chain
from ManimUtils.curve import *
from bitonic import * 
def execute(scene:Scene):
    def create_pixel(mob:Mobject,width:float, height:float)->Group:
        rect = Rectangle(width,height)
        return Group(mob,rect)
    
    
    def part1_animation(start_time :float,single_iteration_time :float):
        end_time = start_time + single_iteration_time
        phase_1_start = remap(0,1,start_time,end_time,0)
        phase_1_end = remap(0,1,start_time,end_time,0.2)
        phase_2_start = remap(0,1,start_time,end_time,0.4)
        phase_2_end = remap(0,1,start_time,end_time,0.6)
        phase_3_start = remap(0,1,start_time,end_time,0.8)
        phase_3_end = remap(0,1,start_time,end_time,1.0)
        
        arr1 = [1,3,7,8]
        arr2 = [17,9,1,-5]
        
        arr1Col1 = Group()
        for i in range(len(arr1)):
            arr1Col1.add(create_pixel(Tex(rf"{arr1[i]}"),0.8,0.8))
        arr1Col1.arrange(direction=DOWN,buff = 0)
        
        
        arr2Col1 = Group()
        for i in range(len(arr2)):
            arr2Col1.add(create_pixel(Tex(rf"{arr2[i]}"),0.8,0.8))
        arr2Col1.arrange(direction=DOWN,buff = 0)
        col1 = Group(arr1Col1,arr2Col1).arrange(direction=DOWN,buff = 1)
        
        
        
        arr1_2 = np.array(arr1, dtype=int)
        arr2_2 = np.array(arr2, dtype=int)
        for i in range(len(arr1)):
            [min0,max0] = [min(arr1[i],arr2[len(arr1) - 1 - i]),max(arr1[i],arr2[len(arr1) - 1 -i])]
            arr1_2[i] = int(max0)
            arr2_2[len(arr1) - 1 - i] = int(min0)
        
        arr1Col2 = Group()
        for i in range(len(arr1)):
            arr1Col2.add(create_pixel(Tex(rf"{arr1_2[i]}"),0.8,0.8))
        arr1Col2.arrange(direction=DOWN,buff = 0)
        arr2Col2 = Group()
        for i in range(len(arr2)):
            arr2Col2.add(create_pixel(Tex(rf"{arr2_2[i]}"),0.8,0.8))
        arr2Col2.arrange(direction=DOWN,buff = 0)
        col2 = Group(arr1Col2,arr2Col2).arrange(direction=DOWN, buff = 1)

        
        arr1Col3 = deepcopy(arr1Col2)
        arr2Col3 = deepcopy(arr2Col2)
        col3 = Group(arr1Col3,arr2Col3).arrange(direction=DOWN,buff = 0)
        
        
        Group(col1,col2,col3).arrange(direction=RIGHT,buff = 3).scale(0.51)
        for i in range(len(arr1)):
            if(arr1[i] == arr1_2[i]):
                keep_obj_move_simplified(arr1Col1[i],arr1Col2[i],phase_1_end,phase_2_start,scene)
            else:
                keep_obj_move_simplified(arr1Col1[i],arr2Col2[len(arr1)-1 - i],phase_1_end,phase_2_start,scene)
            if(arr2[i] == arr2_2[i]):
                keep_obj_move_simplified(arr2Col1[i],arr2Col2[i],phase_1_end,phase_2_start,scene)
            else:
                keep_obj_move_simplified(arr2Col1[i],arr1Col2[len(arr1)-1 - i],phase_1_end,phase_2_start,scene)
        
        keep_obj_move_simplified(arr1Col2,arr1Col3,phase_2_end,phase_3_start,scene)
        keep_obj_move_simplified(arr2Col2,arr2Col3,phase_2_end,phase_3_start,scene)
        keep_obj_alive(col1,phase_1_start,phase_3_end,scene)
        keep_obj_alive(col2,phase_2_start,phase_3_end,scene)
        keep_obj_alive(col3,phase_3_start,phase_3_end,scene)
        return  Group(col1,col2,col3)
        
    def part2_animation(start_time :float,single_iteration_time :float):
        end_time = start_time + single_iteration_time
        phase_1_start = remap(0,1,start_time,end_time,0)
        phase_1_end = remap(0,1,start_time,end_time,0.1)
        phase_2_start = remap(0,1,start_time,end_time,0.3)
        phase_2_end = remap(0,1,start_time,end_time,0.4)
        phase_3_start = remap(0,1,start_time,end_time,0.6)
        phase_3_end = remap(0,1,start_time,end_time,0.7)
        phase_4_start = remap(0,1,start_time,end_time,0.9)
        phase_4_end = remap(0,1,start_time,end_time,1.0)
        
        arr = [17,25,35,42,16,12,-3,-10] 
        steps =  bitonic_sort_steps(arr)
        cols = Group()
        col = Group()
        for entry in arr:
            col.add(create_pixel(Tex(rf"{entry}"),0.9,0.9))
        col.arrange(direction=DOWN,buff = 0)
        cols.add(col)
        for entry in steps:
            col = Group()
            for i in range(len(arr)):
                col.add(create_pixel(Tex(rf"{arr[entry[i]]}"),0.9,0.9))
            col.arrange(direction = DOWN,buff = 0)
            cols.add(col)
        
       
            
        cols.arrange(direction= RIGHT,buff = 2).scale(0.5)
        
        for i, entry in enumerate(cols[0]):
            target = i
            loc = np.where(np.array(steps[0]) == target)[0][0]
            keep_obj_move_simplified(cols[0][i],cols[1][loc],phase_1_end,phase_2_start,scene)
            
        for i, entry in enumerate(cols[1]):
            target = steps[0][i]
            loc = np.where(np.array(steps[1]) == target)[0][0]
            keep_obj_move_simplified(cols[1][i],cols[2][loc],phase_2_end,phase_3_start,scene)
            
        for i, entry in enumerate(cols[2]):
            target = steps[1][i]
            loc = np.where(np.array(steps[2]) == target)[0][0]
            keep_obj_move_simplified(cols[2][i],cols[3][loc],phase_3_end,phase_4_start,scene)
        keep_obj_alive(cols[0],phase_1_start,end_time,scene)
        keep_obj_alive(cols[1],phase_2_start,end_time,scene)
        keep_obj_alive(cols[2],phase_3_start,end_time,scene)
        keep_obj_alive(cols[3],phase_4_start,end_time,scene)
        return cols
            
            
    


    start_tiem = 10
    bitonic_sort_time = 15
    end_time = 30
    
    dotsArrow = []
    for i in range(5):
        dotsArrow.append(Dot(radius=0.3))
        dotsArrow.append(Arrow(ORIGIN,  2 * RIGHT))

    dotsArrow.pop()
    g0 = Group(*dotsArrow).arrange(direction=RIGHT,buff = 0.2)
    def g0_updater(mob,t):
        alpha = remap(0,mob._time,0,1,t) 
        dots0 = mob._dotsArrow
        for i in range(len(dots0)):
            opacity = smoothstep(max(remap(i,i+1,0.,1.,alpha * len(dots0)),0))
            dots0[i].set_opacity(opacity)
        
    g0._dotsArrow = dotsArrow
    g0._time = start_tiem
    keep_obj_alive_update(g0,0,start_tiem,scene,g0_updater)
    
    parallel = {}
    left = 0.3
    right = 0.7
    top = 0.6
    down = 0.4
    width = 5
    height = 4
    parallel["dots"] = []
    parallel["arrows"] = []
    g1 = Group()
    for i in range(width):
        parallel["dots"].append([])
        for j in range(height):
            parallel["dots"][i].append(Dot())
            parallel["dots"][i][j].move_to(uvToManimCoord(
                remap(0,width-1,left,right,i),
                remap(0,height-1,down,top,j)
            ))
            g1.add(parallel["dots"][i][j])
        if(i>=1):
            parallel["arrows"].append([])
            for j in range(height):
       
                    arrow = Arrow(
                            parallel["dots"][i-1][j].get_center(),
                            parallel["dots"][i][j].get_center()
                        )
                    parallel["arrows"][i-1].append(
                        arrow
                    )
                    g1.add(arrow)
    
    g1._elems = parallel
    
    g1._start = start_tiem
    g1._end_time = bitonic_sort_time


    def g1_updater(mob,t):
        alpha = remap(mob._start,mob._end_time,0,1,t)
        dots = mob._elems["dots"]
        arrows = mob._elems["arrows"]
        data = []

        for i in range(len(dots)-1):
            data.append(dots[i])
            data.append(arrows[i])
        data.append(dots[len(dots)-1])
        for i in range(len(data)):
            start = remap(0,len(data),0,1,i)
            end = remap(0,len(data),0,1,i+1)
            opacity = min(max(remap(start,end,0,1,alpha),0),1)
     
            for j in range(len(data[i])):
               
                if(i%2 == 0):
                    data[i][j].scale((0.25 + 0.25 * np.sin(np.pi/2 * opacity))/data[i][j].get_width())
                    data[i][j].set_opacity(0 if opacity < 0.01 else 1)
                else:
                    data[i][j].set_opacity(opacity)
        
        
    g1.scale(2)
    keep_obj_alive_update(g1,start_tiem,bitonic_sort_time,scene,g1_updater)

    
    
            
    

             
            

    one = loop_animation( bitonic_sort_time,end_time+0.5,6,part1_animation)
    two = loop_animation( bitonic_sort_time,end_time+0.5,6,part2_animation)
    
    bitonic_str = Tex(r"Bitonic").move_to(uvToManimCoord(0.75,0.7)).scale(3)
    sort_str = Tex(r"Sort").move_to(uvToManimCoord(0.75,0.3)).scale(3)
    
    keep_obj_alive(bitonic_str,bitonic_sort_time,end_time+0.5,scene)
    keep_obj_alive(sort_str,bitonic_sort_time,end_time+0.5,scene)
    
    def fix_func1(group:Group):

 
        coord = uvToManimCoord(0.25,0.75)
        group.move_to(coord)
    

    loop_animation_fix(one,fix_func1)
    
    def fix_func2(group:Group):
        coord = uvToManimCoord(0.25,0.25)
        group.move_to(coord)
   

    loop_animation_fix(two,fix_func2)
    
  
    
    
    
    
    scene.wait(end_time)

    
   