import re
import moderngl
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
import gc
import time
import subprocess
import os





def save_image_to_series(img: Image.Image, iFrame: int, folder: str = "背景帧序列"):
    # 确保文件夹存在
    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构造文件名，如 frame_0001.png
    filename = output_dir / f"背景_{iFrame:04d}.png"
    
    # 保存图像
    img.save(filename)
    print(f"存储背景第 {iFrame} 帧到: {filename}")

def clear_folder(folder_path: str):
    folder = Path(folder_path)
    if folder.exists() and folder.is_dir():
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()
        print(f"已清空文件夹: {folder}")
    else:
        print(f"文件夹不存在: {folder}")

def load_all_png_images(folder: str = '.') -> list[Image.Image]:
    """
    加载指定文件夹下所有 .png 文件，并返回 Image 对象数组。

    参数:
        folder (str): 要搜索的文件夹路径，默认为当前目录。

    返回:
        list[Image.Image]: 加载的 Image 对象列表。
    """
    path = Path(folder)
    images = []
    for png_file in sorted(path.glob('*.png')):
        try:
            img = Image.open(png_file).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"无法加载 {png_file}: {e}")
    return images

def extract_runtime(filename="背景渲染说明.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        match = re.search(r"运行时间\s*:\s*([\d\.]+)\s*(ms|s)?", content, re.IGNORECASE)
        if match:
            value, unit = match.groups()
            value = float(value)
            if unit == "ms":
                value /= 1000
            print(f"该分镜头运行时间: {value} 秒")
            return value
        else:
            print("未找到运行时间字段")
            return None
    except FileNotFoundError:
        print(f"文件 {filename} 未找到。")
        return None



def preprocess_shader(path: Path, base_dir: Path = None, included=None) -> str:
    """递归替换 GLSL 文件中的 #include 指令"""
    if included is None:
        included = set()

    base_dir = Path("./Utils/")
    code = []
    for line in path.read_text().splitlines():
        if line.strip().startswith("#include"):
            include_name = line.strip().split()[1]
          
            include_path = base_dir / include_name
            
            if include_path in included:
                continue  # 防止循环 include
            included.add(include_path)
            included_code = preprocess_shader(include_path, base_dir, included)
            code.append(f"// Begin include {include_name}")
            code.append(included_code)
            code.append(f"// End include {include_name}")
        else:
            code.append(line)
    return "\n".join(code)

def render_glsl_to_buffer(iFrame: int, iTime: float, imgs: List[Image.Image],vao,fbo) -> Image: 
    resolution=(1280, 720)
    width, height = resolution
    fbo.clear()
    # 设置 uniforms（如果存在）
    prog = vao.program
    if "iResolution" in prog:
        prog["iResolution"].value = (width, height)
    if "iTime" in prog:
        prog["iTime"].value = iTime
    if "iFrame" in prog:
        prog["iFrame"].value = iFrame
    for i, img in enumerate(imgs):
        # 将图像转换为 numpy 数组
        if img is not None and i<3:
            img_data = np.array(img)
            img_data = np.flipud(img_data)
        # 创建一个纹理，绑定到 iChannel1, iChannel2, ... 根据索引 i
            texture = prog.ctx.texture(img.size, 3, img_data.tobytes())
            texture.use(i)  # 绑定纹理到 iChannel[i+1] (GLSL 索引从 1 开始)
            img_data = None

        # 赋值给着色器中的 iChannel1, iChannel2, ... 根据索引 i
            if f"iChannel{i}" in prog:
                prog[f"iChannel{i}"].value = i 

    vao.render()
    # 读取并保存最后一帧为 PNG
    img2 = Image.frombytes('RGB', (width, height), fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)

    return img2

def main(total_time_seconds: float,start_time = 0):
    fps = 30
    """
    模拟时间步进，每次以 1/fps 秒为步长前进，打印每一帧的时间戳和帧编号。

    参数:
    - total_time_seconds: 总运行时间（单位：秒）
    - fps: 每秒帧数（默认30）
    """
    time_step = 1.0 / fps
    total_frames = int(total_time_seconds * fps)
    imgs :List[Image.Image]= [None] * 8
    loaded_images = load_all_png_images()


    for i, img in enumerate(loaded_images):
        imgs[i + 3] = img
 

    ctx = moderngl.create_standalone_context()
    vertices = np.array([
        -1.0, -1.0,   1.0, -1.0,   -1.0, 1.0,
         1.0, -1.0,   1.0,  1.0,  -1.0, 1.0,
    ], dtype='f4')
    vbo = ctx.buffer(vertices)
    fbo = ctx.simple_framebuffer((1280, 720))
    fbo.use()
    vao1 = ctx.simple_vertex_array(
        ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                out vec2 fragUV;
                void main() {
                    fragUV = in_vert * 0.5 + 0.5;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader=preprocess_shader(Path('./buffers/buffer1.glsl'))
        ),
        vbo, 'in_vert'
    )
    vao2 = ctx.simple_vertex_array(
        ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                out vec2 fragUV;
                void main() {
                    fragUV = in_vert * 0.5 + 0.5;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader=preprocess_shader(Path('./buffers/buffer2.glsl'))
        ),
        vbo, 'in_vert'
    )
    vao3 = ctx.simple_vertex_array(
        ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                out vec2 fragUV;
                void main() {
                    fragUV = in_vert * 0.5 + 0.5;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader=preprocess_shader(Path('./buffers/buffer3.glsl'))
        ),
        vbo, 'in_vert'
    )
    vao4 = ctx.simple_vertex_array(
        ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                out vec2 fragUV;
                void main() {
                    fragUV = in_vert * 0.5 + 0.5;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader=preprocess_shader(Path('./frag.glsl'))
        ),
        vbo, 'in_vert'
    )

    for i, img in enumerate(imgs):
        # 将图像转换为 numpy 数组
        
      
        if img is not None and i>=3:
            img_data = np.array(img)
            img_data = np.flipud(img_data)
        # 创建一个纹理，绑定到 iChannel1, iChannel2, ... 根据索引 i

            prog = vao1.program
            texture = prog.ctx.texture(img.size, 3, img_data.tobytes())
            texture.use(i)  # 绑定纹理到 iChannel[i+1] (GLSL 索引从 1 开始)
         

        # 赋值给着色器中的 iChannel1, iChannel2, ... 根据索引 i
            if f"iChannel{i}" in prog:
                prog[f"iChannel{i}"].value = i 
            
            prog = vao2.program
            texture = prog.ctx.texture(img.size, 3, img_data.tobytes())
            texture.use(i)  # 绑定纹理到 iChannel[i+1] (GLSL 索引从 1 开始)
  

        # 赋值给着色器中的 iChannel1, iChannel2, ... 根据索引 i
            if f"iChannel{i}" in prog:
                prog[f"iChannel{i}"].value = i 
            
            prog = vao3.program
            texture = prog.ctx.texture(img.size, 3, img_data.tobytes())
            texture.use(i)  # 绑定纹理到 iChannel[i+1] (GLSL 索引从 1 开始)


        # 赋值给着色器中的 iChannel1, iChannel2, ... 根据索引 i
            if f"iChannel{i}" in prog:
                prog[f"iChannel{i}"].value = i 
            
            prog = vao4.program
            texture = prog.ctx.texture(img.size, 3, img_data.tobytes())
            texture.use(i)  # 绑定纹理到 iChannel[i+1] (GLSL 索引从 1 开始)


        # 赋值给着色器中的 iChannel1, iChannel2, ... 根据索引 i
            if f"iChannel{i}" in prog:
                prog[f"iChannel{i}"].value = i 

    

    res:Image
    start = time.time()
    clear_folder('./背景帧序列')
    for iFrame in range(total_frames):
        if(iFrame%10 ==0):
            gc.collect()
        iTime = (iFrame + 1) * time_step + start_time
        #imgs[0] = render_glsl_to_buffer(iFrame,iTime,imgs,vao1,fbo)
        #imgs[1] = render_glsl_to_buffer(iFrame,iTime,imgs,vao2,fbo)
        #imgs[2] = render_glsl_to_buffer(iFrame,iTime,imgs,vao3,fbo)
        res = render_glsl_to_buffer(iFrame,iTime,imgs,vao4,fbo)
        save_image_to_series(res,iFrame)
        res = None
        print(f"背景第{iFrame + 1}帧渲染完成,当前进度为{iTime/total_time_seconds * 100:.2f}%,帧率为{iFrame/(time.time()-start):.2f}fps")
    print("背景帧序列渲染完成")



       # print(f"Frame {iFrame:04d} - Time: {iTime:.4f}s")
       
def images_to_video(image_folder, image_pattern, fps, output_video):
 
    """
    把图片序列合成为视频
    image_folder: 图片所在文件夹路径，比如 './公式背景帧序列/'
    image_pattern: 图片序列命名格式，如 '公式背景_%d.png'，%d表示数字序号
    fps: 帧率，整数，比如30
    output_video: 输出视频文件名，比如 '分镜头合成视频.mp4'
    """
    # 拼接图片序列的完整路径和格式
    input_path = f"{image_folder}/{image_pattern}"
    
    # ffmpeg参数，注意这里假设图片序号是连续的数字从0或者1开始
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', input_path,
        '-c:v', 'libx264',          # 使用x264编码
        '-pix_fmt', 'yuv420p',     # 兼容大部分播放器
        output_video
    ]
    
    try:
        filename = output_video
        filepath = os.path.join(os.getcwd(), filename)

        if os.path.isfile(filepath):
            os.remove(filepath)
            print(f"已删除文件: {filename}")
        else:
            print(f"文件不存在: {filename}")
        subprocess.run(cmd, check=True)
        print(f"视频合成成功，输出文件: {output_video}")
    except subprocess.CalledProcessError as e:
        print("ffmpeg执行失败:", e)





if __name__ == "__main__":
    runTime = extract_runtime()
    main(runTime,100)


