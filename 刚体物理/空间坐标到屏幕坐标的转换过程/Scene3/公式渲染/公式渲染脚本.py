
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
#好了，我们已经知道了空间中各个点和变量的位置。接下来，我们要把它们放进方程，开始求解。首先，我们有这样一个公式。相机原点P_0，加上步长t，乘以旋转矩阵，再乘以一个列向量。这个向量的三个分量分别是x prime，第二个是数字一，第三个是y prime。这个计算的结果，就是目标点P在三维空间中的坐标。
part1Time = 25
#接下来，我们把相机原点挪到等号右边，变成目标点减去相机原点。这表示目标点相对于相机位置的空间偏移。于是我们有了一个新的表达式。步长t，乘以旋转矩阵，再乘以列向量x prime、一、和y prime，等于目标点减去相机原点。
part2Time = 20
#我们可以把步长t和屏幕坐标合并成一个新的三维向量。这个向量的分量是t乘以x prime，t本身，以及t乘以y prime。在这个向量左边乘以旋转矩阵R，等于目标点减去相机原点。
part3Time = 17
#为了进一步求解这个向量，我们对这个表达式左右两边同时乘以旋转矩阵R的逆矩阵。结果左边变成t乘以列向量，右边是旋转矩阵的逆，乘以目标点减去相机原点。
part4Time = 17
#因为这个三维向量中间的那一项就是t，所以我们可以从中间一项直接解出步长t。具体来说，我们先取行向量零、一、零，也就是只保留中间那一维。用它去乘以旋转矩阵的逆，再乘以目标点和相机原点的差，就得到了步长t的值。
part5Time = 18
#有了t，我们可以将它代入公式，再通过除法操作，计算出屏幕上的投影坐标x prime和y prime。计算方法是。用一个矩阵，这个矩阵第一行是一、零、零，第二行是零、零、一，然后乘以旋转矩阵R的逆，再乘以目标点减去相机原点，最后除以步长t。然后得到的列向量就是屏幕上的投影坐标x prime和y prime，这样，我们就完成了从三维空间坐标到屏幕投影坐标的整个推导过程。
part6Time = 35

class Formula(Scene):
    def construct(self):
    
        f1 = Tex(r"P_0 + t R \left[ \begin{tabular}{c} $x'$ \\ $1$ \\ $y'$ \end{tabular} \right] = P", font_size=30)
        f2 = Tex(r"t R \left[ \begin{tabular}{c} $x'$ \\ $1$ \\ $y'$ \end{tabular} \right] = P - P_0",font_size=30)
        f3 = Tex(r"R \left[ \begin{tabular}{c} $tx'$ \\ $t$ \\ $ty'$ \end{tabular} \right] = P - P_0",font_size=30)
        f4 = Tex(r"\left[ \begin{tabular}{c} $tx'$ \\ $t$ \\ $ty'$ \end{tabular} \right] =R^{-1} (P - P_0)",font_size=30)
        f5 = Tex(r"t = [0,\ 1,\ 0] R^{-1} (P - P_0)",font_size=30)
        f6 = Tex(r"\left[ \begin{array}{c} x' \\[0.1em] y' \end{array} \right]  = \dfrac{1}{t} \left[ \begin{array}{ccc} 1 & 0 & 0 \\ 0 & 0 & 1 \end{array} \right] R^{-1} (P - P_0)",font_size=30)
        f7 = Tex(r"R=[x|y|z]",font_size=80)
        
        f1.move_to(uvToManimCoord([0.6,0.9]),aligned_edge=LEFT)
        f2.move_to(uvToManimCoord([0.6,0.75]),aligned_edge=LEFT)
        f3.move_to(uvToManimCoord([0.6,0.6]),aligned_edge=LEFT)
        f4.move_to(uvToManimCoord([0.6,0.45]),aligned_edge=LEFT)
        f5.move_to(uvToManimCoord([0.6,0.3]),aligned_edge=LEFT)
        f6.move_to(uvToManimCoord([0.6,0.15]),aligned_edge=LEFT)
        f7.move_to(uvToManimCoord([0.35,0.25]))
        


       
        self.add(f7)
        self.play(Write(f1),run_time = part1Time * 0.2)
        self.wait(part1Time * 0.8)
        
        

        self.play(Write(f2),run_time = part2Time * 0.2)
        self.wait(part2Time*0.8)
        
        

        self.play(Write(f3),run_time = part3Time * 0.2)
        self.wait(part3Time*0.8)
        

        self.play(Write(f4),run_time = part4Time * 0.2)
        self.wait(part4Time*0.8)
        

        self.play(Write(f5),run_time = part5Time * 0.2)
        self.wait(part5Time*0.8)
        
   
        self.play(Write(f6),run_time = part6Time * 0.2)
        self.wait(part6Time*0.8)
        



  


















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

    

  

