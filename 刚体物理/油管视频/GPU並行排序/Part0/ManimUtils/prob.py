import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from utils import *
from typing import *

def hash21(x: int, y: int) -> float:
    seed_x = 73856093
    seed_y = 83492791
    hash_value = (x * seed_x) ^ (y * seed_y)
    hash_value = hash_value & 0x7FFFFFFF
    return hash_value / 0x7FFFFFFF


def normal_pdf(x:float,mu:float =0, sigma: float =0.6):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2* sigma ** 2)) 

def create_dynamic_nor()->VGroup:
    axes = Axes(
        x_range=[-2, 2, 1],
        y_range=[-0.3, 1.2, 1],
        axis_config={"include_tip": True}
    )
    curve = axes.get_graph(normal_pdf, x_range= [-1.5,1.5],color=BLUE)
    area = axes.get_area_under_graph(curve, x_range=[-1.5, 1.5], fill_color=RED,fill_opacity=0.3)
    dot = Dot()
    line = Line()

    sigmaTex = Tex(r"\sigma")
    mob = VGroup(axes,curve,area,dot,line,sigmaTex)
    mob.state ={
        "axes":axes,
        "curve":curve,
        "area":area,
        "dot":dot,
        "line":line,
    
        "sigmaTex":sigmaTex
    }
    return mob

def dynamic_nor_updater(mob:VGroup,t:float):
    mu = 0
    sigma = 0.3 + 0.3 * (0.5 + 0.5 * np.sin(2*t))
    def normal_pdf(x:float):
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x-mu)**2 / (2* sigma ** 2)) 
    
    axes = mob.state["axes"]
    curve = mob.state["curve"]
    area = mob.state["area"]
    dot = mob.state["dot"]
    line = mob.state["line"]

    sigmaTex= mob.state["sigmaTex"]
    curve.become(axes.get_graph(normal_pdf, x_range= [-1.5,1.5],color=BLUE))
    area.become(axes.get_area_under_graph(curve, x_range=[-1.5, 1.5], fill_color=RED,fill_opacity=0.3))
    x = np.sin(2 * t) 
    y1 = 0
    y2 = normal_pdf(x)
    point1 = axes.c2p(x,y1)
    point2 = axes.c2p(x,y2)
    dot.become(Dot(point = point2))
    line.become(Line(start = point1, end = point2))

    sigmaTex0 = Tex(rf"\sigma=\sigma_0").move_to(axes.c2p(0,normal_pdf(0)))
    

    sigmaTex.become(sigmaTex0)


    
if __name__ == "__main__":
    print("Utils")