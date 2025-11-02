import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from typing import *
from utils import *
from ManimUtils.animation import *

class Table(Group):
    _rects: List[List[Mobject]]
    _elems: List[List[Mobject]]
    
class Pixel(Group):
    _rect: Rectangle
    _obj: Mobject
    
class Pixels(Group):
    _rects: List[Rectangle]
    _entries: List[Mobject]

def set_mat_elem(mat:Matrix,index: Tuple[int,int],newElem:Tex)->Matrix:
    height = len(mat.get_rows())
    width = len(mat.get_columns())
    entry = mat.get_rows()[index[0]][index[1]] 
    entry.become(newElem.move_to(entry))
    arrs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(get_mat_elem(mat,i,j))
        arrs.append(row)
    return Matrix(arrs)
            
    
def get_mat_elem(mat: Matrix,row:int,col:int)->Mobject:
    return mat.get_row(row)[col]
    

def create_vert_mat()->Matrix:
    mat = Matrix([
        [Tex(r"\vert"),Tex(r"\vert"),Tex(r"\vert")],
        [Tex(r"x"),Tex(r"y"),Tex(r"z")],
        [Tex(r"\vert"),Tex(r"\vert"),Tex(r"\vert")]
    ],
    )
    return mat
    
def create_row_vec(elems)->Matrix:
    vec = Matrix([elems])
    return vec

def create_col_vec(elems:List[any])->Matrix:
    vec = []
    for i, entry in enumerate(elems):
        vec.append([entry])
    vec = Matrix(vec)
    return vec

def get_rect(obj: Mobject)->Rectangle:
    rect = Rectangle(width = 1.2 * obj.get_width(),height = 1.2 * obj.get_height()).move_to(obj)
    return rect

def get_highlight_rect(obj: Mobject)->Rectangle:
    rect = Rectangle(width = 1.2 * obj.get_width(),height = 1.2 * obj.get_height()).move_to(obj).set_opacity(0.5)
    return rect


def create_pixel(obj:Mobject,width:float,height:float)->Pixel:
    rect = Rectangle(width=width,height=height).move_to(obj)
    rect.set_opacity(0)
    res = Group(rect,obj)
    res._obj=obj
    res._rect=rect
    return res

def create_pixels(*objs)->Pixels:
    size = 0
    entries = []
    rects = []
    for entry in objs:
        size = max(size,1.2 * entry.get_width())
        size = max(size,1.2 * entry.get_height())
    mobjs = []
    for entry in objs:
       rect = Rectangle(width = size, height=size)
       entries.append(entry)
       rects.append(rect)
       mobjs.append(Group(rect,entry))
    res = Group(mobjs).arrange(direction=RIGHT,buff = 0)
    res._entries = entries
    res._rects= rects
    return res

def create_table(arrs: List[List[Tex]])->Table:
    width = len(arrs[0])
    height = len(arrs)

    elems = []
    max_elem_width = 0
    max_elem_height = 0
    for i in range(height):
        for j in range(width):
            max_elem_height = max(max_elem_height,arrs[i][j].get_height())
            max_elem_width= max(max_elem_width,arrs[i][j].get_width())
    
    rows = []
    for i in range(height):
        row = []
        for j in range(width):
            pixel = create_pixel(arrs[i][j],max_elem_width,max_elem_height)
            row.append(pixel)
        g = Group(row).arrange(direction=RIGHT,buff = 0.25*max_elem_width)
        rows.append(g)
        temp = []
        for j in range(width):
            temp.append(row[j]._obj)
        elems.append(temp)
    res = Group(rows).arrange(direction=DOWN,buff = 0.25 * max_elem_height)
    res._elems = elems
    res._rects = rows
    return res


def get_table_elem(table:Table,row:int,col:int)->Rectangle:
    return table._elems[row][col]


def get_table_row_elem(table:Table,row:int)->Rectangle:
    width = len(table._rects[0])
    entries = []
    for i in range(width):
        entries.append(table._elems[row][i])
    return Group(entries)

def get_table_col_elem(table:Table,col:int)->Rectangle:
    height = len(table._rects)
    entries = []
    for i in range(height):
        entries.append(table._elems[i][col])
    return Group(entries)

def interchange_mat_rows(
        mat:Matrix,
        row1:int,
        row2:int,

        start_time:float,
        end_time:float,
        scene:Scene
    )->Matrix:
    height = len(mat.get_rows())
    width = len(mat.get_columns())
    old_arrs = []
    for i in range(height):
        row = []
        for j in range(width):
            if i == row1 or i == row2:
                row.append(deepcopy(get_mat_elem(mat,i,j).set_opacity(0)))
            else:  
                row.append(deepcopy(get_mat_elem(mat,i,j)))  
        old_arrs.append(row)
    old_mat = Matrix(old_arrs).move_to(mat)
    
    new_arrs = []
    for i in range(height):
        row = []
        for j in range(width):
            if i == row1:
                row.append(deepcopy(get_mat_elem(mat,row2,j).set_opacity(1)))
            elif i == row2 :
                row.append(deepcopy(get_mat_elem(mat,row1,j).set_opacity(1)))
            else:  
                row.append(deepcopy(get_mat_elem(mat,i,j)))  
        new_arrs.append(row)
    new_mat = Matrix(new_arrs).move_to(old_mat)

    keep_obj_alive(old_mat,start_time,end_time,scene)
    for i in range(width):
        keep_obj_move(deepcopy(get_mat_elem(old_mat,row1,i)).set_opacity(1),
                      deepcopy(get_mat_elem(new_mat,row2,i)).set_opacity(1),
                      start_time,end_time,start_time,end_time,scene)
        keep_obj_move(deepcopy(get_mat_elem(old_mat,row2,i)).set_opacity(1),
                      deepcopy(get_mat_elem(new_mat,row1,i)).set_opacity(1),
                      start_time,end_time,start_time,end_time,scene)
        
  
    return new_mat

def interchange_mat_cols(
        mat:Matrix,
        col1:int,
        col2:int,

        start_time:float,
        end_time:float,
        scene:Scene
    )->Matrix:
    height = len(mat.get_rows())
    width = len(mat.get_columns())
    old_arrs = []
    for i in range(height):
        row = []
        for j in range(width):
            if j == col1 or j == col2:
                row.append(deepcopy(get_mat_elem(mat,i,j).set_opacity(0)))
            else:  
                row.append(deepcopy(get_mat_elem(mat,i,j)))  
        old_arrs.append(row)
    old_mat = Matrix(old_arrs).move_to(mat)
    
    new_arrs = []
    for i in range(height):
        row = []
        for j in range(width):
            if j == col1:
                row.append(deepcopy(get_mat_elem(mat,i,col2).set_opacity(1)))
            elif j == col2 :
                row.append(deepcopy(get_mat_elem(mat,i,col1).set_opacity(1)))
            else:  
                row.append(deepcopy(get_mat_elem(mat,i,j)))  
        new_arrs.append(row)
    new_mat = Matrix(new_arrs).move_to(old_mat)

    keep_obj_alive(old_mat,start_time,end_time,scene)
    for i in range(height):
        keep_obj_move(deepcopy(get_mat_elem(old_mat,i,col1)).set_opacity(1),
                      deepcopy(get_mat_elem(new_mat,i,col2)).set_opacity(1),
                      start_time,end_time,start_time,end_time,scene)
        keep_obj_move(deepcopy(get_mat_elem(old_mat,i,col2)).set_opacity(1),
                      deepcopy(get_mat_elem(new_mat,i,col1)).set_opacity(1),
                      start_time,end_time,start_time,end_time,scene)
        
  
    return new_mat
    
 
        
    # keep_obj_alive(mat,start_time,end_time,scene)





def get_pixels_elem(pixels:Pixels,idx:int)->Mobject:
    return pixels._entries[idx]

def get_pixels_rect(pixels:Pixels,idx:int)->Rectangle:
    return pixels._rects[idx]