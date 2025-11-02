import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from utils import *
from typing import *
import numpy as np
from pathlib import *
import moderngl
from concurrent.futures import ThreadPoolExecutor

#shader名字
TUNNEL = "tunnel"
RAYTRACING = "rayTracing"


def transform_mainImage_to_main(code: str) -> str:
    # 匹配整个 mainImage 函数签名和开头大括号
    pattern = r"void\s+mainImage\s*\([^)]*\)\s*\{"
    match = re.search(pattern, code)


    if not match:
        return code  # 没匹配到就原样返回

    start, end = match.span()
    
    # 构造新的函数头部
    new_header = "void main() {\n    vec2 fragCoord = vec2(gl_FragCoord.xy);"


    # 替换旧头部
    return code[:start] + new_header + code[end:]


def review_text(path:Path)->str:
    common_file_path = Path(os.path.join(os.path.dirname(path),"common.glsl"))
    code1 = common_file_path.read_text()
    code1 = transform_mainImage_to_main(code1)
    code2 = path.read_text()
    code2 = transform_mainImage_to_main(code2)
    code = code1 + code2
    code = """
        #version 330 core
        uniform vec2      iResolution;   
        uniform float     iTime;       
        uniform int       iFrame;      
        uniform sampler2D iChannel0;
        uniform sampler2D iChannel1;
        uniform sampler2D iChannel2;
        uniform sampler2D iChannel3;
        uniform sampler2D iChannel4;
        uniform sampler2D iChannel5;
        uniform sampler2D iChannel6;
        uniform sampler2D iChannel7;
        out vec4 fragColor;
    """ + code
    return code

def get_code_by_name(proj_name:str,file_name:str)->str:
    dir = find_nearest_dir_with("shaders")
    file_path = Path(os.path.join(dir,f'{proj_name}/{file_name}.glsl'))
    return review_text(file_path)

def basic_standard_width_tansform(height:float)->float:
    return 1280/720*height


def get_textures(names:List[str]|None = None)->List[Image.Image]|None:
    if(names == None): 
        return None 
    else:
        imgs = []
        for name in names:
            dir_path = find_nearest_dir_with("assets")
            file_path = os.path.join(dir_path,f"shader_texture/{name}.png")
            imgs.append(Image.open(file_path))
        return imgs

def create_renderer(proj_name:str,file_name:str, width:float, height:float):
    frag_shader_code = get_code_by_name(proj_name=proj_name,file_name=file_name)
    ctx = moderngl.create_standalone_context()

    prog = ctx.program(
        vertex_shader='''
            #version 330
            in vec2 in_vert;
            out vec2 fragCoord;
            void main() {
                fragCoord = in_vert * 0.5 + 0.5;
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        ''',
        fragment_shader=frag_shader_code
    )

    if 'iResolution' in prog:
        prog['iResolution'].value = (width, height)

    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')
    fbo = ctx.simple_framebuffer((width, height))
    fbo.use()

    return {
        'ctx': ctx,
        'program': prog,
        'vao': vao,
        'fbo': fbo,
        'width': width,
        'height': height,
    }
    


def render_frame(renderer, time: float, frame: int, textures:List[Image.Image|None]|None=None):
    import PIL.Image
    prog = renderer['program']
    ctx = renderer['ctx']
    fbo = renderer['fbo']
    vao = renderer['vao']

    def pil_to_texture(pil_img: PIL.Image.Image):
        # 确保图像格式为RGBA或RGB，方便上传
        if pil_img.mode not in ('RGB', 'RGBA'):
            pil_img = pil_img.convert('RGBA')
        
        # 翻转图像垂直方向，OpenGL纹理坐标与PIL坐标相反
        pil_img = pil_img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        img_data = pil_img.tobytes()
        width, height = pil_img.size
        components = len(pil_img.mode)  # 3 or 4
        texture = ctx.texture((width, height), components, data=img_data)
        texture.build_mipmaps()
        return texture

    # === 设置 uniforms ===
    if 'iTime' in prog:
        prog['iTime'].value = time
    if 'iFrame' in prog:
        prog['iFrame'].value = frame
    if 'iResolution' in prog:
        prog['iResolution'].value = (renderer['width'], renderer['height'])

    # === 纹理绑定 ===
    if textures:
        for i, tex in enumerate(textures):
            if i >= 4:
                break
            if tex == None:
                return
            else:
                tex = pil_to_texture(tex)
                channel_name = f"iChannel{i}"
                prog[channel_name].value = i
                tex.use(location=i)
    # === 渲染 ===
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    vao.render()

    raw_data = fbo.read(components=3, alignment=1)
    img = PIL.Image.frombytes("RGB", (renderer['width'], renderer['height']), raw_data)
    img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    return img

def generate_standard_image(size:int)->Tuple[np.ndarray,Image.Image]:
    width = 128
    height = int(np.ceil(size / width))
    data = np.zeros((height, width, 4), dtype=np.uint8)
    arr = data
    img = Image.fromarray(data, mode="RGBA")
    return (arr,img)



def write_num_to_image(arr:np.ndarray,idx:int,num:float):
    width = 128
    row = int(np.floor(idx/width))
    col = int(idx%width)
    arr[row, col] = list(encode_float(num)) 
    
    

    
    
    


def make_param_texture(nums: List[float]) -> Image.Image:
    width = 128
    count = len(nums)
    # 计算需要多少行：首行第一个像素存count，剩余空间每像素存一个float
    # 第一像素占用1个，后面每个数占1个像素
    total_pixels = count 
    height = int(np.ceil((total_pixels) / width))  # 向上取整行数

    # 创建空白numpy数组，初始化为0，形状（height, width, 4），uint8格式
    data = np.zeros((height, width, 4), dtype=np.uint8)



    # 从第1个像素开始存数字
    for idx, num in enumerate(nums):
        px = list(encode_float(num))
 
        # 计算二维坐标
        pos = idx 
        row = np.floor(pos/width)
        col = pos % width
        for i in range(4):
            data[int(row),int(col),int(i)] = int(px[i])
        # data[row, col] = px

    # 转换为PIL图像
    img = Image.fromarray(data, mode="RGBA")
  
    return img


            

def basic_render(
        source_name:str,
        target_name:str,
        width:float,
        height:float,
        run_time:float,
        textures:List[str] = None,
        executor:ThreadPoolExecutor|None = None
    ):
    textures = get_textures(textures)
    renderer = create_renderer(
        proj_name=source_name,
        file_name="main",
        width=int(width),
        height=int(height)
    )
    t=0
    frame = 0
    if(frame > 0): textures = None
    while t < run_time:
        img = render_frame(renderer=renderer,time=t,frame = frame,textures=textures)
        if(executor!=None):
            executor.submit(img_saver,img,target_name,frame)
        else:
            img_saver(img,target_name,frame)
        t+=1/30
        frame+=1
        
def compu_shift_render(
        source_name:str,
        target_name:str,
        width:float,
        height:float,
        run_time:float,
        executor:ThreadPoolExecutor|None = None
    ):

    renderer = create_renderer(
        proj_name="compuShift",
        file_name="main",
        width=int(width),
        height=int(height)
    )
    t=0
    frame = 0
    pixel =encode_float(run_time)  # 例如：紫蓝色像素


# 创建一个 1x1 的图像
    img2 = Image.new("RGBA", (1, 1), pixel)
    if(frame > 0): textures = None
    while t < run_time:
        source_img_path = get_frame_path_str(source_name,frame,False)
        textures = [Image.open(source_img_path)]
        if(frame == 0):
            textures.append(img2)
   
            
        img = render_frame(renderer=renderer,time=t,frame = frame,textures=textures)
        if(executor!=None):
            executor.submit(img_saver,img,target_name,frame)
        else:
            img_saver(img,target_name,frame)
        t+=1/30
        frame+=1
        
        
