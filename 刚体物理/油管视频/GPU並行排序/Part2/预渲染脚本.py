import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from ManimUtils.utils import *
import subprocess
from manimlib import *
from typing import *

if __name__ == "__main__":
    source_dir = find_nearest_dir_with("场景渲染模版")
    for filename in os.listdir(source_dir):
        if filename.startswith("pir"):
            str_path = str(os.path.join(source_dir,filename))
            subprocess.run(["python3",str_path])



