import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from utils import *
from typing import *
from animation import * 


def write_mobj(mob:Mobject,start_time:float,write_end_time:float,end_time:float,scene:Scene):
    tex_copy = deepcopy(mob)
    if(write_end_time<=start_time): return
    len0 = len(list(mob.submobjects))
    if(len0 == 0): return
    for i,entry in enumerate(mob.submobjects):
        delta_time = (write_end_time - start_time)/len0
        cur_start_time = remap(0,len0-1,start_time,start_time + delta_time * (len0-1),i)
        cur_end_time = remap(1,len0,start_time+delta_time,write_end_time,i+1)
        keep_obj_fade_in(entry,cur_start_time,cur_end_time,scene)
    keep_obj_invisable(mob,end_time,scene)
  
    



def words2Paragraph(words:str,width:float)->Group:
    words = words.split()
    flag = 0
    arrs=[]
    str0 = ""
    print(len(words))
    while(flag<len(words)):
        tempstr = f"{words[flag]}"
        if (Tex(rf"{str0+tempstr}").get_width()>width):
            temp = list(tempstr)
            str1 = ""
            str2 = ""
            for i in range(len(temp)):
                str2 = "".join(temp[i+1:])+"\\ "
                str1 = str0 + "".join(temp[:i+1])
                if(Tex(rf"{str0+"".join(temp[:i+1])}").get_width()>width):
                    break
            arrs.append(Tex(rf"{str1}"))
            str0 = str2
        else:
            str0 = str0 + f"{tempstr}\\ "
        
        if(flag == len(words) - 1):
            arrs.append(Tex(rf"{str0}"))
        flag = flag+1
    return Group(arrs).arrange(direction=DOWN,aligned_edge= LEFT)

def write_paragraph(words:str,width:float,start_time:float,end_write_time:float,end_time:float,scene:Scene)->Group:
    para = words2Paragraph(words,width)
    rows = []
    widths = []
    total_width = 0
    for entry in para:
        rows.append(entry)
        total_width+=entry.get_width()
        widths.append(total_width)
    for i, entry in enumerate(para):
        if i == 0:
            cur_start_time = remap(0,1,start_time,end_write_time,0)
            cur_end_time = remap(0,1,start_time,end_write_time,widths[0]/total_width)
            write_mobj(entry,cur_start_time,cur_end_time,end_time,scene)
        else:
            cur_start_time = remap(0,1,start_time,end_write_time,widths[i-1]/total_width)
            cur_end_time = remap(0,1,start_time,end_write_time,widths[i]/total_width)
            write_mobj(entry,cur_start_time,cur_end_time,end_time,scene)
    return para
