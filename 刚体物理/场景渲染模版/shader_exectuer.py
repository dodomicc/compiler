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

       
def raytrace_render(
        target_name:str,
        width:float,
        height:float,
        run_time:float,
        roughness_func:Callable|None = None,
        executor:ThreadPoolExecutor|None = None
    ):
    
    renderer = create_renderer(
        proj_name=RAYTRACING,
        file_name="main",
        width=int(width),
        height=int(height)
    )
    t = 0
    frame  = 0

    while t < run_time:
        textures = []
        if(roughness_func!=None):
            textures.append(make_param_texture([roughness_func(t)]))
        img = render_frame(renderer=renderer,time=t,frame = frame,textures=textures)
        if(executor!=None):
            executor.submit(img_saver,img,target_name,frame)
        else:
            img_saver(img,target_name,frame)
        t+=1/30
        frame+=1

        
