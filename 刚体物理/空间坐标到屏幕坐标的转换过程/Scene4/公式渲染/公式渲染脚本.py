
import subprocess
import os
import cv2
import shutil
from manimlib import *
from typing import List


resolution = [1280,720]
max_frame_width = 0.5 *  resolution[0]/resolution[1] * manim_config["sizes"]["frame_height"]
max_frame_height = manim_config["sizes"]["frame_height"]/2



#需要操作的公式部分的主要函数

def remap(val1,val2,val3,val4,val):
    h = (val - val1)/(val2 - val1)
    res = val3 + h * (val4 - val3)
    return res 


#将uv坐标转化为manim坐标
def uvToManimCoord(uv):
    res = [(2 * uv[0] - 1)*max_frame_width, (2 * uv[1] - 1)*max_frame_height, 0]
    return res



#公式位置更新函数
def updater(mob, alpha):
    time = remap(0,1,1.2,4.1,alpha)
    start = LEFT * 3
    end = RIGHT * 3
    mob.move_to(interpolate(start, end, time))

partTime = 17

class Formula(Scene):
    def construct(self):
    
 
        
        f1 = Tex(r"P_0 = (x_0,\, y_0,\, z_0)",font_size=30)
        f2= Tex(r"\text{yAxis} = \mathrm{normalize}(Q - P_0)",font_size=30)
     
        f3= Tex(
            r"\text{xAxis} = \text{normalize} \left( \text{yAxis} \times \left[ \begin{array}{c} 0 \\ 1 \\ 0 \end{array} \right] \right)",
            font_size=30
        )
        f4 = Tex(
            r"\text{zAxis} = \text{normalize}\left( \text{xAxis} \times \text{yAxis} \right)",
            font_size=30
        )
        f5 = Tex(
            r"R=[xAxis|yAxis|zAxis]",
            font_size=30
        )
        f6 = Tex(
            r"P = (x,\, y,\, z)",
            font_size=30
        )
        f7 = Tex(
            r"t = || P - P_0 ||",
            font_size=30
        )
        
        f8 = Tex(r"P_0 + t R \left[ \begin{tabular}{c} $x'$ \\ $1$ \\ $y'$ \end{tabular} \right] = P", font_size=30)
        f9 = Tex(r"t R \left[ \begin{tabular}{c} $x'$ \\ $1$ \\ $y'$ \end{tabular} \right] = P - P_0",font_size=30)
        f10 = Tex(r"R \left[ \begin{tabular}{c} $tx'$ \\ $t$ \\ $ty'$ \end{tabular} \right] = P - P_0",font_size=30)
        f11 = Tex(r"\left[ \begin{tabular}{c} $tx'$ \\ $t$ \\ $ty'$ \end{tabular} \right] =R^{-1} (P - P_0)",font_size=30)
        f12 = Tex(r"t = [0,\ 1,\ 0] R^{-1} (P - P_0)",font_size=30)
        f13 = Tex(r"\left[ \begin{array}{c} x' \\[0.1em] y' \end{array} \right]  = \dfrac{1}{t} \left[ \begin{array}{ccc} 1 & 0 & 0 \\ 0 & 0 & 1 \end{array} \right] R^{-1} (P - P_0)",font_size=30)
        
        f1.move_to(uvToManimCoord([0.2,0.85-1]),aligned_edge=LEFT)
        f2.move_to(uvToManimCoord([0.2,0.75-1]),aligned_edge=LEFT)
        f3.move_to(uvToManimCoord([0.2,0.65-1]),aligned_edge=LEFT)
        f4.move_to(uvToManimCoord([0.2,0.55-1]),aligned_edge=LEFT)
        f5.move_to(uvToManimCoord([0.2,0.45-1]),aligned_edge=LEFT)
        f6.move_to(uvToManimCoord([0.2,0.35-1]),aligned_edge=LEFT)
        f7.move_to(uvToManimCoord([0.2,0.25-1]),aligned_edge=LEFT)
        
        
        f8.move_to(uvToManimCoord([0.6,0.9-1]),aligned_edge=LEFT)
        f9.move_to(uvToManimCoord([0.6,0.75-1]),aligned_edge=LEFT)
        f10.move_to(uvToManimCoord([0.6,0.6-1]),aligned_edge=LEFT)
        f11.move_to(uvToManimCoord([0.6,0.45-1]),aligned_edge=LEFT)
        f12.move_to(uvToManimCoord([0.6,0.3-1]),aligned_edge=LEFT)
        f13.move_to(uvToManimCoord([0.6,0.15-1]),aligned_edge=LEFT)
        
        group = VGroup(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13)
        print(group)

        
        self.add(f1)
        self.add(f2)
        self.add(f3)
        self.add(f4)
        self.add(f5)
        self.add(f6)
        self.add(f7)
        self.add(f8)
        self.add(f9)
        self.add(f10)
        self.add(f11)
        self.add(f12)
        self.add(f13)
        self.play(group.animate.shift(UP * 7.6),run_time = partTime * 0.4)

        self.wait(partTime * 0.6)

  


















def extract_frames_from_formula_video(fps=30):
    video_path = os.path.join("media", "公式.mp4")
    output_dir = "公式帧序列"

    if not os.path.exists(video_path):
        print(f"找不到视频文件：{video_path}")
        return

    # 如果输出目录已存在，则清空
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件:", video_path)
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频帧率: {video_fps}fps, 应该生成总帧数: {total_frames}")
  

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = os.path.join(output_dir, f"公式_{saved:04d}.png")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"完成全部视频帧序列提取：共提取 {saved} 帧，保存在 “{output_dir}” 文件夹中。")

        
      

def main():
    # 构建命令

    file_path = os.path.join("media", "公式.mp4")

# 检查文件是否存在
    if os.path.exists(file_path):
        os.remove(file_path)
        print("已删除 公式.mp4")
    else:
        print("未找到 公式.mp4")
    subprocess.run([
    "manimgl",
    "公式渲染脚本.py",
    "Formula",
    "-w",
    "-m",
    "--video_dir", "./media",
    "--file_name", "公式.mp4"
    "--quiet"
    "-t"
    ])

    print(f"已生成公式视频文件:公式.mp4")
    extract_frames_from_formula_video()



if __name__ == "__main__":
    main()

    

  

