import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from ManimUtils.utils import *
import subprocess
from manimlib import *
from ManimUtils.basic_2d_scenes import *
from typing import *
from ManimUtils.glsl_render import *
import shutil


from main import *


class Formula(Scene):
    def construct(self):
        #定义场景时间
        execute(self)



        
      

def main():
    # 构建命令
    file_path = os.path.join(find_nearest_dir_with("media"), "可視化展示視頻.mp4")

# 检查文件是否存在
    if os.path.exists(file_path):
        os.remove(file_path)
        print("已删除 可視化展示視頻.mp4")
    else:
        print("未找到 可視化展示視頻.mp4")
    subprocess.run([
    "manimgl",
    "场景渲染脚本.py",
    "Formula",
    "-w",
    "-m",
    "--video_dir", "./media",
    "--file_name", "可視化展示視頻.mp4"
    "--quiet"
    "-t"
    ])

    print(f"已生成视频文件:可視化展示視頻.mp4")


def is_bitonic(arr):
    trend_changes = 0
    increasing = None
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            if increasing is False:
                trend_changes += 1
                increasing = True
            elif increasing is None:
                increasing = True
        elif arr[i] < arr[i - 1]:
            if increasing is True:
                trend_changes += 1
                increasing = False
            elif increasing is None:
                increasing = False
        if trend_changes > 1:
            return False
    return True

if __name__ == "__main__":
    print("開始渲染")   
    # num = 15
    # arr1 =  np.random.rand(num)
    # arr2 =  np.random.rand(num)
    # arr1 = 2 * (arr1 - 0.5) * 50
    # arr2 = 2 * (arr2 - 0.5) * 50
    # arr1 = np.sort(arr1)
    # arr2 = np.sort(arr2)[::-1]

    
    # res = np.zeros(2 * num)
    # for i in range(num):
    #     res[i] = max(arr1[i],arr2[num-1-i])
    #     res[2 * num - 1 - i] = min(arr1[i],arr2[num-1-i])
        
    
    # print(res)
    # print("Bitonic?" , is_bitonic(res))


    
   
    
    main()
    dir1 = find_nearest_dir_with("media")
    dir2 = find_nearest_dir_with("视频声音合并")
    source_file_path = os.path.join(Path(dir1),"可視化展示視頻.mp4")
    dist_file_path = os.path.join(Path(dir2),"Scene.mp4")
    if(os.path.exists(source_file_path)):
        shutil.copy(str(source_file_path),str(dist_file_path))
    else:
        print("不存在文件-可視化展示視頻.mp4")

    


  


