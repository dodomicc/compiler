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

from PhysicalEngine.constraint import *
import jax.numpy as jnp
import shutil
from main import *
import matplotlib.pyplot as plt
from PhysicalEngine.sdf2Voxels_2d import * 
from PhysicalEngine.utils import *
from PhysicalEngine.convexHull import *
from compute_shader.shader_drive import *
from Jaco_Grad.Jaco_Grad import *

from linalg.linalg import *





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



if __name__ == "__main__":
    print("開始渲染")   
    # size =150
    # arr = np.zeros(size,dtype=np.float32)
    # for i in range(size):
    #     arr[i] = np.random.rand() *3000.
    # arr = jnp.array(arr)
    # loop_angle_constraint_grad = jit(grad(smooth_loop_angle_constraint_2d))
    # loop_length_constraint_grad = jit(grad(smooth_loop_length_constraint_2d))
    # ballon_constraint_grad = jit(grad(self_ballon_constraint_2d))
    # points = arr.reshape(-1,2)
    # fig, axs = plt.subplots(1, 3, figsize=(10, 2))
    # for i in range(3):  
    #     axs[i].set_xticks([])
    #     axs[i].set_yticks([])
    # points = jnp.vstack([points,points[0]])
    # axs[0].plot(points[:,0],points[:,1],'-')
    # axs[0].axis('equal')
    # def updater(point:jnp.array,i:int,max_iter:int):
    #     alpha = i/max_iter
    #     penalty_scaler = 200. * smoothstep(remap(0.,0.6,0.,1.,alpha)) 
    #     self_intersection_penalty_scaler = 500. * smoothstep(remap(0,0.4,0,1,alpha)) 
    #     expansion_decay_rate = 0.0000001 * (1. + smoothstep(remap(0.6,0.8,0.,1.,alpha)) * 100000.)
    #     grad_angle =  loop_angle_constraint_grad(point,penalty_scaler)
    #     grad_angle = jnp.where(jnp.isnan(grad_angle), 0., grad_angle)
    #     grad_length =  loop_length_constraint_grad(point,0.,penalty_scaler)
    #     grad_length = jnp.where(jnp.isnan(grad_length), 0., grad_length)
    #     if(i%(int(np.ceil(0.2 * max_iter)))== 0 ):
    #         print(alpha)
    #         points = point.reshape(-1,2)
    #         points = jnp.vstack([points,points[0]])
    #         if(i!=0):
    #             axs[1].plot(points[:,0],points[:,1],'-')
    #             axs[1].axis('equal')
    #     grad_ballon =  ballon_constraint_grad(point,expansion_decay_rate, self_intersection_penalty_scaler)
    #     grad_ballon = jnp.where(jnp.isnan(grad_ballon), 0., grad_ballon)
    #     return point - 0.004 * grad_ballon - 0.0001 * grad_angle - 0.0001 * grad_length 
    # arr = step0(arr,updater,5000)
    # points = arr.reshape(-1,2)
    # points = jnp.vstack([points,points[0]])
    # axs[2].plot(points[:,0],points[:,1],'-')
    # axs[2].axis('equal')
    # plt.show()
    # def decode_float(bytes_tuple: Tuple[int, int, int, int]) -> float:
    #     return struct.unpack('f', struct.pack('4B', *bytes_tuple))[0]
    # num = np.random.rand()
    # arr = num * np.ones((4,3),dtype=np.float32)
    # arr = arr.reshape(1,-1).view(np.uint8).view(np.float32).reshape((4,3))
    # print(num)
    size = 48
    A = np.random.rand(size,size) * 1000 - 500
    times = 10

    time1 = time.time()
    for i in range(times):
        qr(A)
        # scipy.linalg.qr(A)
    time2 = time.time()
   

    print("qr分解檢驗")
    print(f"單次{size}*{size}隨機矩陣qr分解耗時")
    print(f"{round((time2 - time1)/times * 1000,2)}ms")
    q,r = qr(A)
    low_tri = np.tril(r)
    np.fill_diagonal(low_tri,0.)
    print("上三角檢驗")
    print(np.allclose(low_tri,np.zeros(size)))
    print("正交性檢驗")
    print(np.allclose(q.T@q , np.identity(size)))
    print("還原性檢驗")
    print(np.allclose(q@r ,A))
    
    
    # symmetry_mat = A.T @ A
    # time1 = time.time()
    # for i in range(times):
    #     jacobi_diag(symmetry_mat)
    # time2 = time.time()
    # print("------------------------------------------------------")
  
    # diag , orth_mat =jacobi_diag(symmetry_mat)

    # diag_test = diag.copy()
    
    # np.fill_diagonal(diag_test, 0.)
    # print("對稱矩陣對角化分解檢驗")
    # print(f"單次{size}*{size}對稱矩陣對角化分解耗時")
    # print(f"{round((time2 - time1)/times * 1000,2)}ms")
    # print("對角化檢驗")
    # print(np.allclose(diag_test,np.zeros((size,size))))
    # print("正交性檢驗")
    # print(np.allclose(orth_mat.T @ orth_mat, np.identity(size)))
    # print("還原性檢驗")
    # print(np.allclose(orth_mat@ diag @ orth_mat.T, symmetry_mat))
    
    
    

    # main()
    # dir1 = find_nearest_dir_with("media")
    # dir2 = find_nearest_dir_with("视频声音合并")
    # source_file_path = os.path.join(Path(dir1),"可視化展示視頻.mp4")
    # dist_file_path = os.path.join(Path(dir2),"Scene.mp4")
    # if(os.path.exists(source_file_path)):
    #     shutil.copy(str(source_file_path),str(dist_file_path))
    # else:
    #     print("不存在文件-可視化展示視頻.mp4")




    


  


