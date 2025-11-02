import os
import shutil
import subprocess
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import moderngl
import numpy as np

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


def clear_folder(folder_path):
    """
    删除指定文件夹下的所有文件（保留文件夹）
    folder_path: 文件夹路径，例如 './公式背景帧序列'
    """
    if not os.path.exists(folder_path):
        print(f"目录不存在: {folder_path}")
        return

    file_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                file_count += 1
        except Exception as e:
            print(f"删除文件失败: {file_path}，错误: {e}")
    
    print(f"已删除 {file_count} 个文件。")


def run_scripts():
    # Step 1: 分别在各自目录中执行渲染脚本
    subprocess.run(["python3", "公式渲染脚本.py"], check=True, cwd="./公式渲染")
    subprocess.run(["python3", "背景渲染脚本.py"], check=True, cwd="./背景渲染")

def collect_png_frames():
    # Step 2: 合并 PNG 文件到当前目录下的“公式背景帧序列”
    src_dirs = ["./公式渲染/公式帧序列", "./背景渲染/背景帧序列"]
    dest_dir = "./公式背景帧序列"
    os.makedirs(dest_dir, exist_ok=True)

    for src in src_dirs:
        for fname in os.listdir(src):
            if fname.lower().endswith(".png"):
                full_src_path = os.path.join(src, fname)
                full_dest_path = os.path.join(dest_dir, fname)
                shutil.copy2(full_src_path, full_dest_path)

def clear_original_frame_folders():
    # Step 3: 删除源目录中的所有文件和子文件夹（不删除目录本身）
    frame_dirs = ["./公式渲染/公式帧序列", "./背景渲染/背景帧序列"]
    for dir_path in frame_dirs:
        for item in os.listdir(dir_path):
            full_path = os.path.join(dir_path, item)
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)


def blend_images(background_path, overlay_path, output_path,vao,fbo):
    # 打开背景和公式图像，确保为RGBA模式
    background = Image.open(background_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB")

    # 图像尺寸应相同
    if background.size != overlay.size:
        raise ValueError(f"尺寸不一致: {background_path} 和 {overlay_path}")
 

    # 获取像素数据

    
    fbo.clear()
    # 设置 uniforms（如果存在）
    prog = vao.program
    #print("Uniforms:", prog._members.keys())
    if "iResolution" in prog:
        prog["iResolution"].value = (1280, 720)
    img_data = np.array(background)  
        # 创建一个纹理，绑定到 iChannel1, iChannel2, ... 根据索引 i
    texture = prog.ctx.texture(background.size, 3, img_data.tobytes())
    texture.use(0)  

    prog["iChannel0"].value = 0
    img_data = np.array(overlay)
        # 创建一个纹理，绑定到 iChannel1, iChannel2, ... 根据索引 i
    texture = prog.ctx.texture(overlay.size, 3, img_data.tobytes())
    texture.use(1)  
    prog["iChannel1"].value = 1
    img_data = None

    vao.render()
    background = Image.frombytes('RGB', (1280, 720), fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)
    



    # 保存新图像
    background.save(output_path)



def merge_all_images_in_folder(folder_path):
    def process_images(start_i, step,vao,fbo):
        i = start_i
        while True:
            bg_path = os.path.join(folder_path, f"背景_{i:04d}.png")
            fg_path = os.path.join(folder_path, f"公式_{i:04d}.png")
            out_path = os.path.join(folder_path, f"公式背景_{i}.png")

            if not os.path.exists(bg_path) or not os.path.exists(fg_path):
                break

            print(f"🔧 合成：{out_path}")
            blend_images(bg_path, fg_path, out_path,vao,fbo)
            i += step
    
    ctx = moderngl.create_standalone_context()
    vertices = np.array([
        -1.0, -1.0,   1.0, -1.0,   -1.0, 1.0,
         1.0, -1.0,   1.0,  1.0,  -1.0, 1.0,
    ], dtype='f4')
    vbo = ctx.buffer(vertices)
    fbo = ctx.simple_framebuffer((1280, 720))
    fbo.use()
    vao = ctx.simple_vertex_array(
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
            fragment_shader=preprocess_shader(Path('./视频合成.glsl'))
        ),
        vbo, 'in_vert'
    )






    process_images(0, 1,vao,fbo)
    print("✅ 所有图像合成完毕")


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
        

def copy_video( 
                                  source_file='./分镜头合成视频.mp4', 
                                  target_dir='../视频声音合并'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.basename(current_dir)
    # 构建目标路径
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f'{folder_name}.mp4')

    # 执行复制
    shutil.copy2(source_file, target_path)
    print(f'✅ 复制完成：{target_path}')




if __name__ == "__main__":
    run_scripts()
    collect_png_frames()
    clear_original_frame_folders()
    merge_all_images_in_folder("./公式背景帧序列")
    images_to_video(
        image_folder='./公式背景帧序列',
        image_pattern='公式背景_%d.png',
        fps=30,
        output_video='分镜头合成视频.mp4'
    )
    clear_folder('./公式背景帧序列')
    copy_video()
    print("✅ 全部执行完毕")