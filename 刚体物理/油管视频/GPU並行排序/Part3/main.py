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
    part1_res = 3
    part2_res = 12
    phase_1_start = 20
    phase_1_end = 23
    phase_2_start = 25
    phase_2_end = 27
    phase_3_start = 29
    phase_3_end = 31
    phase_4_start = 33
    phase_4_end = 35
    phase_5_start = 37
    phase_5_end = 39
    phase_6_start = 41
    phase_6_end = 43.5
    
    end_time = phase_6_end - 0.5
    
    start = Group(Tex(r"Part \ 3"), Tex(r"Sort \ a \ Sequence \ of \ Length \ 2^k")).arrange(direction=RIGHT,buff = 1).scale(1.5)
    keep_obj_alive(start,0,part1_res,scene)
    
    summary_top = Group(
        Tex(r"Part \ 1:").scale(1.5),
        Tex(r"an \ increasing \ and \ a \ decreasing \ sequence \ can \ form \ a \ Bitonic \ Sequence").scale(0.8),

        ).arrange(direction=DOWN,aligned_edge = LEFT)
    arrow =  Arrow(ORIGIN,2.5 * RIGHT)
    summary_bottom =Group(
        Tex(r"Part \ 2:").scale(1.5),
        Group(
        Tex(r"Bitonic \ Sequence") ,
        arrow,
        Tex(r"increasing \ or \ decreasing \ sequence")
        
        ).arrange(direction=RIGHT).scale(0.8)
        ).arrange(direction=DOWN,aligned_edge = LEFT)
    
    Group(summary_top,summary_bottom).arrange(direction=DOWN,aligned_edge = LEFT,buff = 2)
    
    summary_bottom.add(Tex(r"Batcher \ Merge").scale(0.7).move_to(arrow.get_center() + UP * arrow.get_height(),aligned_edge=DOWN))
    
    
    keep_obj_alive(summary_top,part1_res,phase_1_start,scene)
    keep_obj_alive(summary_bottom,part2_res,phase_1_start,scene)
    
    
    def create_pixel(mob:Mobject,width:float, height:float)->Group:
        rect = Rectangle(width,height)
        return Group(mob,rect)
    
    first_layer = []
    for i in range(8):
        first_layer.append(create_pixel(Tex(rf"a_{i}"),1.2,0.8))
    first_layer_group = Group(*first_layer).arrange(direction=RIGHT)
    
    second_layer = []
    for i in range(4):
        elem = Group(first_layer[2 * i],first_layer[2 * i+1])
        width = elem.get_width()
        height =elem.get_height()
        if(i%2 == 0):
            second_layer.append(create_pixel(Tex(r"increasing"),width,height).move_to(elem))
        else:
            second_layer.append(create_pixel(Tex(r"decreasing"),width,height).move_to(elem))
    second_layer_group = Group(*second_layer)
    
    third_layer = []
    for i in range(2):
        elem = Group(first_layer[4 * i],first_layer[4 * i + 1], first_layer[4 * i + 2],first_layer[4 * i + 3])
        width = elem.get_width()
        height =elem.get_height()
        third_layer.append(create_pixel(Tex(r"Bitonic"),width,height).move_to(elem))
    third_layer_group = Group(*third_layer)
    Group(first_layer_group,second_layer_group,third_layer_group).arrange(direction=DOWN)
    
    forth_layer = []
    for i in range(2):
        elem = Group(first_layer[4 * i],first_layer[4 * i + 1], first_layer[4 * i + 2],first_layer[4 * i + 3])
        width = elem.get_width()
        height =elem.get_height()
        if i == 0:
            forth_layer.append(create_pixel(Tex(r"increasing"),width,height).move_to(elem))
        else:
            forth_layer.append(create_pixel(Tex(r"decreasing"),width,height).move_to(elem))
    forth_layer_group = Group(*forth_layer)
    
    fifth_layer_group = create_pixel(Tex(r"Bitonic"),first_layer_group.get_width(),first_layer_group.get_height()).move_to(first_layer_group)
    sixth_layer_group = create_pixel(Tex(r"increasing"),first_layer_group.get_width(),first_layer_group.get_height()).move_to(first_layer_group)
    Group(
        first_layer_group,
        second_layer_group,
        third_layer_group,
        forth_layer_group,
        fifth_layer_group,
        sixth_layer_group
        ).arrange(direction=DOWN)
    

    for i in range(8):
        keep_obj_move_with_fade(first_layer[i],second_layer[int(np.floor(i/2))],phase_1_end,phase_2_start,phase_1_end,phase_2_start,scene)
    for i in range(4):
        keep_obj_move_with_fade(second_layer[i],third_layer[int(np.floor(i/2))],phase_2_end,phase_3_start,phase_2_end,phase_3_start,scene)
    for i in range(2):
        keep_obj_move_with_fade(third_layer[i],forth_layer[i],phase_3_end,phase_4_start,phase_3_end,phase_4_start,scene)
    for i in range(2):
        keep_obj_move_with_fade(forth_layer_group[i],fifth_layer_group,phase_4_end,phase_5_start,phase_4_end,phase_5_start,scene)
    keep_obj_move_with_fade(fifth_layer_group,sixth_layer_group,phase_5_end,phase_6_start,phase_5_end,phase_6_start,scene)
    
    keep_obj_alive(first_layer_group,phase_1_start,phase_6_end,scene)
    keep_obj_alive(second_layer_group,phase_2_start,phase_6_end,scene)
    keep_obj_alive(third_layer_group,phase_3_start,phase_6_end,scene)
    keep_obj_alive(forth_layer_group,phase_4_start,phase_6_end,scene)
    keep_obj_alive(fifth_layer_group,phase_5_start,phase_6_end,scene)
    keep_obj_alive(sixth_layer_group,phase_6_start,phase_6_end,scene)
    
    


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
        
        
        Group(col1,col2,col3).arrange(direction=RIGHT,buff = 3).scale(0.3)
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
        
       
            
        cols.arrange(direction= RIGHT,buff = 2).scale(0.3)
        
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
            
            
                
            

    one  = loop_animation(part1_res,phase_1_start,6,part1_animation)
    two =  loop_animation(part2_res,phase_1_start,6,part2_animation)
    def fix_func1(group:Group):

        width = group.get_width()
        height = group.get_height()
        coord = [
            4-width/2,
            summary_top[1].get_center()[1]+summary_top[1].get_height()/2+height/2,
            
            0]
        group.move_to(coord)
    loop_animation_fix(one,fix_func1)
    
    def fix_func2(group:Group):

        width = group.get_width()
        height = group.get_height()
        coord = [
            4-width/2,
            summary_bottom[1].get_center()[1]+summary_bottom[1].get_height()/2+height/2,
            
            0]
        group.move_to(coord)
    loop_animation_fix(two,fix_func2)

    # two = loop_animation(55,65,7,part2_animation)
    
    
    
    
    scene.wait(end_time)

    
   