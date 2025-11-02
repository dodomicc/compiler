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
def execute(scene:Scene):
    batcher_theorem_start = 5 
    example_time = 8
    first_compare = 10
    second_compare = 13
    third_compare = 16
    forth_compare = 19
    batcher_merge_explain = 22
    example_end_time = 25
    col2_start_time = 30
    col3_start_time = 35
    
    end_time = 40
    start = Group(Tex(r"Part \ 2 :"),Tex(r"Sort \ Bitonic \ Sequence")).arrange(direction=RIGHT,buff = 0.25).scale(2)
    keep_obj_alive(start,0,batcher_theorem_start,scene)
    batcher_theorem = Group(
        Tex(r"Batcher \  Theorem"),
        Tex(r"Given \ a \ bitonic \  sequence \ of \ length \ 2^k,it \ can \  be \ sorted \ into \ a \ monotonic "),
        Tex(r"sequence \ sing \ a \ fixed \ sequence \ of \ comparison-exchange \ operations. "),
        Tex(r"This \ sorting \ process, \ known \ as \ Batcher's \ bitonic \ merge"),
    ).arrange(direction=DOWN,aligned_edge = LEFT).scale(0.8)
    keep_obj_alive(batcher_theorem,batcher_theorem_start,example_time,scene)
    
    seq = [15, 23, 12, 10, 9, 6, 4, 1]
    for i in range(len(seq)):
        seq[i] = Tex(rf"{seq[i]}")
    seq = create_pixels(*seq).scale(2)
    
    curve1 = get_curve(seq._rects[0].get_center(),seq._rects[4].get_center(),scale=0.8, color=GOLD)
    curve2 = get_curve(seq._rects[1].get_center(),seq._rects[5].get_center(),scale=0.8, color=RED)
    curve3 = get_curve(seq._rects[2].get_center(),seq._rects[6].get_center(),scale=0.8, color=GREEN)
    curve4 = get_curve(seq._rects[3].get_center(),seq._rects[7].get_center(),scale=0.8)
    
    keep_obj_alive(seq,example_time,example_end_time,scene)
    keep_obj_alive(curve1,example_time,batcher_merge_explain,scene)
    keep_obj_alive(curve2,example_time,batcher_merge_explain,scene)
    keep_obj_alive(curve3,example_time,batcher_merge_explain,scene)
    keep_obj_alive(curve4,example_time,batcher_merge_explain,scene)
    
    
    keep_obj_move2(seq._rects[0],seq._rects[4],seq._entries[0],example_time,example_end_time,first_compare,second_compare,scene)
    keep_obj_move2(seq._rects[4],seq._rects[0],seq._entries[4],example_time,example_end_time,first_compare,second_compare,scene)
    
    keep_obj_move2(seq._rects[1],seq._rects[5],seq._entries[1],example_time,example_end_time,second_compare,third_compare,scene)
    keep_obj_move2(seq._rects[5],seq._rects[1],seq._entries[5],example_time,example_end_time,second_compare,third_compare,scene)
    
    keep_obj_move2(seq._rects[2],seq._rects[6],seq._entries[2],example_time,example_end_time,third_compare,forth_compare,scene)
    keep_obj_move2(seq._rects[6],seq._rects[2],seq._entries[6],example_time,example_end_time,third_compare,forth_compare,scene)
    
    keep_obj_move2(seq._rects[3],seq._rects[7],seq._entries[3],example_time,example_end_time,forth_compare,batcher_merge_explain,scene)
    keep_obj_move2(seq._rects[7],seq._rects[3],seq._entries[7],example_time,example_end_time,forth_compare,batcher_merge_explain,scene)
    
    
    rect1 = get_rect(Group(seq._rects[0],seq._rects[1],seq._rects[2],seq._rects[3])).scale(1/1.2)
    rect1.set_color(GREEN)
    rect2 = get_rect(Group(seq._rects[4],seq._rects[5],seq._rects[6],seq._rects[7])).scale(1/1.2)
    rect2.set_color(RED)
    keep_obj_flash(rect1,batcher_merge_explain,example_end_time,batcher_merge_explain,example_end_time,scene)
    keep_obj_flash(rect2,batcher_merge_explain,example_end_time,batcher_merge_explain,example_end_time,scene)
    
    
    def generate_Box(n:int)->Group:
        box = Group(Tex(r"Bitonic \ Sequence"),Tex(rf"length = {n}")).arrange(direction=DOWN).scale(0.6)
        rect = get_rect(box)
        return Group(box,rect)
    col1 = generate_Box(8)
    
    col2_box1 = generate_Box(4)
    col2_box2 = generate_Box(4)
    col2 = Group(col2_box1,Tex("<=").rotate(3 * np.pi/2),col2_box2).arrange(direction=DOWN,buff = 1.5)
    Group(col1,col2).arrange(direction=RIGHT)
    
    col3_box1 = generate_Box(2)
    col3_box2 = generate_Box(2)
    col3_box3 = generate_Box(2)
    col3_box4 = generate_Box(2)
    col3 = Group(
        col3_box1,
        Tex("<=").rotate(3 * np.pi/2),
        col3_box2,
        Tex("<=").rotate(3 * np.pi/2),
        col3_box3,Tex("<=").rotate(3 * np.pi/2),
        col3_box4
        ).arrange(direction=DOWN)
    Group(col1,col2,col3).arrange(direction=RIGHT,buff = 1.5)
    col2.add(
        Arrow(col1.get_center(),col2_box1.get_center()),
             Arrow(col1.get_center(),col2_box2.get_center())
            )
    col3.add(
        Arrow(col2_box1.get_center(),col3_box1.get_center()),
             Arrow(col2_box1.get_center(),col3_box2.get_center()),
             Arrow(col2_box2.get_center(),col3_box3.get_center()),
             Arrow(col2_box2.get_center(),col3_box4.get_center())
    )
    keep_obj_alive(col1,example_end_time,end_time+1,scene)
    keep_obj_alive(col2,col2_start_time,end_time+1,scene)
    keep_obj_alive(col3,col3_start_time,end_time+1,scene)
    
    
    
    


    


    
    
    
    scene.wait(end_time)

    
   