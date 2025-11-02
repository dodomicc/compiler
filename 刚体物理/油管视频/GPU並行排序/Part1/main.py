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
def execute(scene:Scene):

    def_time = 3
    example_1_time = 8
    example_2_time = 12
    seqs_time = 20
    first_elem_flash = 22
    second_elem_flash = 25
    first_elem_mov = 28
    other_elem_move = 32
    seqs_move = 37
    end_time = 40
    
    
    # 雙調定義部分
    bitonic__def_1 = Tex(r"A\ sequence \ x_0, x_1, \dots, x_{n-1}\ is\ bitonic")
    bitonic__def_2 = Tex(r"if\ there\ exists \ k \ such \ that:")
    bitonic__def_3 = Tex(r"x_0<=x_1<=\cdots<=x_k>=x_{k+1}>=\cdots>=x_{n_1}")
    bitonic__def = Group(bitonic__def_1,bitonic__def_2,bitonic__def_3).arrange(direction=DOWN,aligned_edge = LEFT)
   
    
    #雙調定義例子
    row1 = [2, 5, 8, 12, 10, 7, 3]
    bitonic_example_1_vec = create_row_vec(row1)
    bitonic_example_1_axes = Axes(x_range=[0,13,1],
                                 y_range=[0,12,1],
                             
                                axis_config={"include_tip": True}
                                ).scale(0.3)
    bitonic_example_1 = Group(bitonic_example_1_vec,bitonic_example_1_axes).arrange(direction=RIGHT)
    dots = []
    arrows = []
    for i in range(len(row1)):
        point = bitonic_example_1_axes.c2p(i*2,row1[i])
        dots.append(Dot(point))
        if(i>=1): arrows.append(Arrow(start = dots[i-1].get_center(),end = dots[i].get_center()))
    bitonic_example_1.add(*(dots + arrows))
    row2 =  [3, 7, 5, 9, 2]
    bitonic_example_2_vec = create_row_vec(row2)
    bitonic_example_2_axes = Axes(x_range=[0,12,1],
                                 y_range=[0,12,1],
                             
                                axis_config={"include_tip": True}
                                ).scale(0.3)
    bitonic_example_2 = Group(bitonic_example_2_vec,bitonic_example_2_axes).arrange(direction=RIGHT,buff = 2.9)
    dots2 = []
    arrows2 = []
    for i in range(len(row2)):
        point = bitonic_example_2_axes.c2p(i*2,row2[i])
        dots2.append(Dot(point))
        if(i>=1): arrows2.append(Arrow(start = dots2[i-1].get_center(),end = dots2[i].get_center()))
    bitonic_example_2.add(*dots2,*arrows2)
    Group(bitonic_example_1,bitonic_example_2).arrange(direction=DOWN,aligned_edge = LEFT)
    rect1 = get_rect(bitonic_example_1_vec)
    rect2 = get_rect(bitonic_example_2_vec)
    # 展现定义
    keep_obj_alive(bitonic__def,
                   def_time,
                   example_1_time
                   ,scene)
    # 展现例子1
    keep_obj_alive(bitonic_example_1,example_1_time,seqs_time,scene)
    keep_obj_flash(rect1,example_1_time,example_2_time,example_1_time,example_2_time,scene)
    #展现例子2
    keep_obj_alive(bitonic_example_2,example_2_time,seqs_time,scene)
    keep_obj_flash(rect2,example_2_time,seqs_time,example_2_time,seqs_time,scene)
    
    convert_start = Group(Tex(r"Part \ 1").scale(2.5), 
                          Group(
                              Tex(r"Two \ Monotonic \ Sequences"),
                              Arrow([0,0,0],[0,-1,0]),
                              Tex(r"Bitonic \ Sequence")
                              ).arrange(direction=DOWN).scale(1.5)
                          ).arrange(direction= RIGHT,buff = 1)
    convert_inc_seq = [1,3,5,7,13,13,14,16,19,45]
    convert_de_seq = [32,27,27,15,10,9,5,5,3,2]
    convert_inc_arr = []
    convert_de_arr = []
    for i in range(10):
        convert_inc_arr.append(Tex(f"{convert_inc_seq[i]}"))
        convert_de_arr.append(Tex(f"{convert_de_seq[i]}"))
    convert_inc_seq = create_pixels(*convert_inc_arr)
    convert_de_seq = create_pixels(*convert_de_arr)
    seqs = Group(convert_inc_seq,convert_de_seq).arrange(direction=DOWN,buff = 4)
    
    convert_inc_arr = [1,3,5,7,13,13,14,16,19,45]
    convert_de_arr =  [32,27,27,15,10,9,5,5,3,2]

    #標註對偶元素
    dual_rect_1 = Group(convert_inc_seq._rects[0],convert_de_seq._rects[9])
    dual_rect_2 = Group(convert_inc_seq._rects[1],convert_de_seq._rects[8])
    
    # 本幕開場
    write_mobj(convert_start,0,def_time,def_time,scene)
    
    
    keep_obj_alive(seqs,seqs_time,seqs_move,scene)
    
    #閃爍對偶元素
    keep_obj_flash(dual_rect_1,first_elem_flash,seqs_move,first_elem_flash,second_elem_flash,scene)
    keep_obj_flash(dual_rect_2,second_elem_flash,seqs_move,second_elem_flash,first_elem_mov,scene)
    
    #移動第一個對偶元素

    keep_obj_move2(convert_inc_seq._rects[0],convert_de_seq._rects[9],convert_inc_seq._entries[0],first_elem_mov,seqs_move,first_elem_mov,other_elem_move,scene)
    keep_obj_move2(convert_de_seq._rects[9],convert_inc_seq._rects[0],convert_de_seq._entries[9],first_elem_mov,seqs_move,first_elem_mov,other_elem_move,scene)
    temp = convert_inc_arr[0]
    convert_inc_arr[0] =  convert_de_arr[9]

    convert_de_arr[9] = temp
  
    

    
    # 移動其他對偶元素
    for i in range(9):
        if(convert_inc_arr[i+1]<convert_de_arr[9 - (i+1)]):
            keep_obj_move2(convert_inc_seq._rects[i+1],convert_de_seq._rects[8 - i],convert_inc_seq._entries[i+1],other_elem_move,seqs_move,other_elem_move,seqs_move,scene)
            keep_obj_move2(convert_de_seq._rects[8-i],convert_inc_seq._rects[i+1],convert_de_seq._entries[8 - i],other_elem_move,seqs_move,other_elem_move,seqs_move,scene)
            temp = convert_inc_arr[i+1]
            convert_inc_arr[i+1] =  convert_de_arr[8-i]
    
            convert_de_arr[8-i] =  temp
       
    
    # 加載數組場景
    
    
    convert_inc_seq = convert_inc_arr
    convert_de_seq = convert_de_arr
    convert_inc_arr = []
    convert_de_arr = []
    for i in range(10):
        convert_inc_arr.append(Tex(f"{convert_inc_seq[i]}"))
        convert_de_arr.append(Tex(f"{convert_de_seq[i]}"))
    convert_inc_seq = create_pixels(*convert_inc_arr)
    convert_de_seq = create_pixels(*convert_de_arr)
    Group(convert_inc_seq,convert_de_seq).arrange(direction=DOWN,buff = 4)
    convert_inc_seq2 = deepcopy(convert_inc_seq).move_to(convert_inc_seq)
    convert_de_seq2 = deepcopy(convert_de_seq).move_to(convert_de_seq)
    Group(convert_inc_seq2,convert_de_seq2).arrange(direction=RIGHT,buff = 0)
    
    
    # 讓兩個數列移到指定位置
    keep_obj_move(convert_inc_seq,convert_inc_seq2,seqs_move,end_time+1,seqs_move,end_time,scene)
    keep_obj_move(convert_de_seq,convert_de_seq2,seqs_move,end_time+1,seqs_move,end_time,scene)
    



    
    
    
        
    
    

    scene.wait(end_time)

    
   