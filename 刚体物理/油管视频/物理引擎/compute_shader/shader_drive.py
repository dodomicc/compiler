import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from ManimUtils.utils import *
from typing import *
import numpy as np
from pathlib import *
import moderngl
from pathlib import Path

ALGOPREFIX = "prefix_sum"
ALGOMINVALIDX = "min_val_idx"
ALGOMAXVALIDX = "max_val_idx"
ALGOCOMPARE = "compare"
ALGOCONDITION = "condition"
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


def combine_code(custom_part_name:str|None =None,algorithm_part_name:str|None = None)->str:
    path = os.path.join(os.getcwd(),"compute_shader/shaders")
    
    # 导入通用模块
    common_path = Path(os.path.join(path,"utils/common.glsl"))
    code_common = common_path.read_text()

    matrix_path = Path(os.path.join(path,"utils/matrix.glsl"))
    code_matrix = matrix_path.read_text()
    
    grad_path = Path(os.path.join(path,"utils/grad.glsl"))
    code_grad = grad_path.read_text()
    
    
    # 导入定制算法模块
    custom_part = ""
    if(custom_part_name!=None):
        custom_part_path = Path(os.path.join(path,f"{custom_part_name}.glsl"))
        custom_part = custom_part_path.read_text()
    
    algorithm_part = ""
    if(algorithm_part_name!= None):
        algorithm_part_path = Path(os.path.join(path,f"algorithm/{algorithm_part_name}.glsl"))
        algorithm_part =  algorithm_part_path.read_text()
    # 导入主模块
    code_main_path = Path(os.path.join(path,"utils/main.glsl"))
    code_main=transform_mainImage_to_main(code_main_path.read_text())
    
    code = code_common + code_matrix + code_grad + custom_part+ algorithm_part + code_main
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


class ShaderController:
    def __init__(self, width: int):

        self.width = width
        self.ctx = moderngl.create_standalone_context()
        self.progs:dict = {}
        self.textures:dict = {}
        self.node_outputs: dict = {}  
        self.fbos:dict = {} 
        self.node_states:dict = {} 
        self.vaos: dict = {}  
        self.setup_vao()
        
    def setup_vao(self):
        self.vertices = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
            1.0, -1.0,
            1.0,  1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(self.vertices.tobytes())

    def add_program(self,prog_name: str,width:int,height:int, custom_part_name:str|None = None, algorithm_part_name:str|None = None):
        # 加载 fragment shader
        if not prog_name in self.progs:
            prog = self.ctx.program(
                vertex_shader='''
                    #version 330
                    in vec2 in_vert;
                    out vec2 fragCoord;
                    void main() {
                        fragCoord = in_vert * 0.5 + 0.5;
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                    }
                ''',
                fragment_shader=combine_code(custom_part_name,algorithm_part_name)
            )
    
    
            self.progs[prog_name] = prog
            self.vaos[prog_name] = self.ctx.simple_vertex_array(prog, self.vbo, 'in_vert')
            height = int(np.ceil(width * height/self.width) + 1.)   


            tex0 = self.ctx.texture((self.width, height), 4, dtype='f4')
            tex1 = self.ctx.texture((self.width, height), 4, dtype='f4')
            self.node_outputs[prog_name] = [tex0, tex1]


            fbo0 = self.ctx.framebuffer(color_attachments=[tex0])
            fbo1 = self.ctx.framebuffer(color_attachments=[tex1])
            self.fbos[prog_name] = [fbo0,fbo1]
            
            self.node_states[prog_name] = {'cur_idx': 0}
    
            
    def bind_inputs(self, prog_name: str, textures: List[moderngl.Texture]):
        prog = self.progs[prog_name]
        for i, tex in enumerate(textures):
            tex.use(location = i)
            prog[f"iChannel{i}"] = i


    
    def get_output_texture(self, prog_name: str):
        return self.node_outputs[prog_name][1- self.node_states[prog_name]['cur_idx']]
    
    
    def write_texture(self, arr:np.ndarray,name:str)->moderngl.Texture:
        arr1 = np.zeros(arr.shape[0] * arr.shape[1] + 2)
        arr1[0] = arr.shape[0]
        arr1[1] = arr.shape[1]
        arr1[2:] = arr.flatten()       
        if(name in self.textures): 
            width ,height1 = self.textures[name].size
            height = int(np.ceil(len(arr1)/(4 * width)) ) + 1
            arr1_total_len = 4 * int(max(height,height1) * width)
            arr1 = np.concatenate([arr1,np.zeros(arr1_total_len - len(arr1))]).astype(np.float32)
            if(height1>=height):
                tex:moderngl.Texture = self.textures[name]
                tex.write(arr1)
                return tex
            else:
                tex = self.ctx.texture((width,height),4,dtype='f4')
                self.textures[name] = tex
                tex.write(arr1)
                return tex
        else:
            width = self.width
            height = int(np.ceil(len(arr1)/(4 * width)) ) + 1
            arr1_total_len = 4 * int(height * width)
            arr1 = np.concatenate([arr1,np.zeros(arr1_total_len - len(arr1))]).astype(np.float32)
            tex = self.ctx.texture((width,height),4,dtype='f4')
            self.textures[name] = tex
            tex.write(arr1)
            return tex
        
      

    
    
    def get_arr_from_texture(self,texture:moderngl.Texture)->np.ndarray:
        arr = np.frombuffer(texture.read(),dtype=np.float32)
        arr1 = arr[2:int(arr[0] * arr[1]) + 2]
        arr2 = arr1.reshape((int(arr[0]), int(arr[1])))
        return arr2
    
    
    def render_node(self, prog_name: str, frame_id: int, textures: List[moderngl.Texture] = []):
        state:int = self.node_states[prog_name]
        idx = state['cur_idx'] 
        tex_out = self.node_outputs[prog_name][idx]
        fbo = self.fbos[prog_name][idx]
        fbo.use()

        prog: moderngl.Program= self.progs[prog_name]
        prog['iResolution'].value = tex_out.size
        prog['iFrame'].value = frame_id
        textures.insert(0,self.get_output_texture(prog_name))
        self.bind_inputs(prog_name,textures)
        self.vaos[prog_name].render()

        # 切换 ping-pong
        state['cur_idx'] = 1 - state['cur_idx']