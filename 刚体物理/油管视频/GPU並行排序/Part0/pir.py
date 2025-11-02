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
from shader_exectuer import * 
from physics import *

if __name__ == "__main__":
   print("開始預渲染")