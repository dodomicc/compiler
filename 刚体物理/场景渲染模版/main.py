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
    t = Tex(r"A~sequence~$\{x_0, x_1, \dots, x_{n-1}\}$~is~bitonic~if~there~exists~$k$~such~that:",
    r"$$x_0 \le x_1 \le \dots \le x_k \ge x_{k+1} \ge \dots \ge x_{n-1}$$",
    r"or",
    r"$$x_0 \ge x_1 \ge \dots \ge x_k \le x_{k+1} \le \dots \le x_{n-1}$$")
    scene.add(t)
    scene.wait(20)

    
   