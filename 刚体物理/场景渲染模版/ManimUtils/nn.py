import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from utils import *
from itertools import * 

def create_nn(width:float,height:float)->Group:
    dots = []
    lines = []
    linecols = []
    for i in range(5):
        cols = []
        x = remap(0,4,-width/2,width/2,i)
        if i == 0:
            for j in range(5):
                y = 0.6 * remap(0,4,-height/2,height/2,j)
                dot = Dot([x,y,0],0.1)
                cols.append(dot)
        elif i<4:
            for j in range(8):
                y = remap(0,7,-height/2,height/2,j)
                dot = Dot([x,y,0],0.1)
                cols.append(dot)
        else:
            for j in range(3):
                y = 0.4 * remap(0,2,-height/2,height/2,j)
                dot = Dot([x,y,0],0.1)
                cols.append(dot)
        dots.append(cols)
        
    for i,entry in enumerate(dots):
        if i<len(dots) - 1:
            lincol = []
            for start in entry:
                for end in dots[i+1]:
                    line = Line(start.get_center(),end.get_center())
                    lincol.append(line)
                    lines.append(line)
            linecols.append(lincol)
    elems = list(chain.from_iterable(dots)) + lines          
    nn = Group(*elems)
    nn._dots = dots
    nn._lines = lines
    rand = []
    start = []
    for line in lines:
        rand.append(np.random.rand() * 5 + 1)
        start.append(np.random.rand() * 5)
    nn._rand = rand
    nn._start = start
    nn._linecols = linecols
    return nn

def nn_updater(mob,t):
    for i, line in enumerate(mob._lines):
        line.set_opacity(np.abs(np.sin(mob._start[i]+ mob._rand[i]*t )))